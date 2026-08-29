import asyncio
import shlex
import sys

import pytest

from torchtyc.cli import _overrides_from, build_parser
from torchtyc.config import Overrides
from torchtyc.diagnostics import Severity
from torchtyc.mux import (
    QUEUE_LIMIT,
    Downstream,
    Mux,
    Pending,
    _deep_merge,
    _encode,
    _hover_text,
    _read_message,
    torchtyc_command,
)


def merge(method: str, results: list) -> object:
    pending = Pending(client_id=1, method=method, expected=len(results), results=results)
    return Mux([]).merge(pending)


def test_initialize_merges_capabilities():
    merged = merge(
        "initialize",
        [
            {"capabilities": {"hoverProvider": True, "workspace": {"a": 1}}},
            {"capabilities": {"codeLensProvider": {}, "workspace": {"b": 2}}},
        ],
    )
    caps = merged["capabilities"]
    assert caps["hoverProvider"] is True
    assert "codeLensProvider" in caps
    assert caps["workspace"] == {"a": 1, "b": 2}


def test_list_results_concatenate():
    assert merge("textDocument/codeAction", [[1, 2], [3]]) == [1, 2, 3]


def test_none_results_are_dropped():
    assert merge("textDocument/codeLens", [None, [1]]) == [1]
    assert merge("textDocument/definition", [None, None]) is None


def test_hovers_stack():
    merged = merge(
        "textDocument/hover",
        [
            {"contents": {"kind": "markdown", "value": "from pyright"}},
            {"contents": {"kind": "markdown", "value": "from torchtyc"}},
        ],
    )
    assert "from pyright" in merged["contents"]["value"]
    assert "from torchtyc" in merged["contents"]["value"]


def test_semantic_tokens_delta_is_passed_through_unwrapped():
    delta = {"resultId": "7", "edits": [{"start": 0, "deleteCount": 1, "data": [1, 2, 3]}]}
    assert merge("textDocument/semanticTokens/full/delta", [delta]) == delta


def test_other_methods_take_the_first_answer():
    assert merge("textDocument/rename", [{"changes": {}}, {"changes": {"x": 1}}]) == {"changes": {}}


def test_ids_are_unique_per_server():
    mux = Mux([])
    assert mux.next_id() != mux.next_id()


def test_hover_text_forms():
    assert _hover_text("plain") == "plain"
    assert _hover_text({"value": "v"}) == "v"
    assert _hover_text([{"value": "a"}, "b"]) == "a\n\nb"
    assert _hover_text(None) == ""


def test_deep_merge_keeps_first_scalar():
    into = {"a": 1}
    _deep_merge(into, {"a": 2, "b": 3})
    assert into == {"a": 1, "b": 3}


def test_deep_merge_unions_lists():
    into = {"a": [1, 2]}
    _deep_merge(into, {"a": [2, 3]})
    assert into["a"] == [1, 2, 3]


def reader_with(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


def test_end_of_stream_raises_eof():
    async def run():
        with pytest.raises(EOFError):
            await _read_message(reader_with(b""))

    asyncio.run(run())


def test_unparseable_body_is_skipped_not_fatal():
    async def run():
        reader = reader_with(
            b"Content-Length: 3\r\n\r\nnot"
            + _encode({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        )
        assert await _read_message(reader) is None
        assert (await _read_message(reader))["method"] == "initialized"

    asyncio.run(run())


def test_a_malformed_frame_does_not_stop_the_pump():
    seen: list[dict] = []

    class Recorder(Mux):
        async def from_client(self, message):
            seen.append(message)

    async def run():
        reader = reader_with(
            b"Content-Length: 0\r\n\r\n"
            + b"Content-Length: 3\r\n\r\nnot"
            + _encode({"jsonrpc": "2.0", "method": "exit"})
        )
        await Recorder([]).pump_client(reader)

    asyncio.run(run())
    assert [m["method"] for m in seen] == ["exit"]


def child_overrides(overrides: Overrides) -> Overrides:
    argv = shlex.split(torchtyc_command(overrides))
    return _overrides_from(build_parser().parse_args(argv[argv.index("lsp") :]))


def test_mux_forwards_its_options_to_the_child_server():
    overrides = Overrides(
        python="/tmp/venv/bin/python",
        variadic_rank=4,
        ignore=frozenset({"unused-dim"}),
        severity=Severity.WARNING,
        timeout=5.0,
    )
    assert child_overrides(overrides) == overrides


def test_mux_forwards_nothing_the_user_did_not_ask_for():
    assert child_overrides(Overrides()) == Overrides()


async def _stalled_server(name: str = "stalled"):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    return Downstream(name=name, process=process)


async def _reading_server(name: str = "reading"):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; sys.stdin.buffer.read()",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    return Downstream(name=name, process=process)


def test_a_stalled_server_is_dropped_instead_of_stalling_the_client_loop():
    """One server that stops reading must not hold up the others."""

    async def run():
        stalled = await _stalled_server()
        reading = await _reading_server()
        mux = Mux([])
        mux.servers = [stalled, reading]
        writers = [asyncio.create_task(s.pump_writes()) for s in mux.servers]
        note = {"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {"x": "y" * 4096}}
        try:
            # More frames than one server's queue holds, so the stalled server
            # must be given up on rather than blocking the loop.
            async def feed():
                for _ in range(QUEUE_LIMIT + 50):
                    await mux.from_client(dict(note))
                    # Let the reading server finish each frame, so what fills up
                    # is only ever the queue of the server that stopped reading.
                    await reading.outbox.join()

            await asyncio.wait_for(feed(), timeout=30.0)
            # The stalled server never read a byte, yet the loop ran to the end
            # and its neighbour took every frame.
            assert stalled.alive is False
            assert reading.alive is True
            assert reading.outbox.empty()
        finally:
            for task in writers:
                task.cancel()
            for server in mux.servers:
                if server.process.returncode is None:
                    server.process.kill()
                await server.process.wait()

    asyncio.run(run())


def test_a_request_already_sent_to_a_dropped_server_still_answers_the_client():
    """The client must not wait forever for a server the mux gave up on."""

    async def run():
        stalled = await _stalled_server()
        mux = Mux([])
        mux.servers = [stalled]
        answered: list[dict] = []

        async def capture(message):
            answered.append(message)

        mux.to_client = capture  # type: ignore[method-assign]
        writer = asyncio.create_task(stalled.pump_writes())
        try:
            for index in range(QUEUE_LIMIT + 50):
                await mux.from_client(
                    {
                        "jsonrpc": "2.0",
                        "id": index,
                        "method": "textDocument/hover",
                        "params": {"blob": "z" * 4096},
                    }
                )
            assert stalled.alive is False
            # Every request the mux accepted has an answer, error or otherwise.
            assert len(answered) == QUEUE_LIMIT + 50
            assert mux.pending == {}
        finally:
            writer.cancel()
            stalled.process.kill()
            await stalled.process.wait()

    asyncio.run(run())
