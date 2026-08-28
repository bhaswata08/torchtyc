import asyncio
import shlex

import pytest

from torchtyc.cli import _config_from, build_parser
from torchtyc.config import Config
from torchtyc.diagnostics import Severity
from torchtyc.mux import (
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


def test_mux_forwards_its_options_to_the_child_server(tmp_path):
    config = Config(
        root=tmp_path,
        python="/tmp/venv/bin/python",
        variadic_rank=4,
        ignore=frozenset({"unused-dim"}),
        severity=Severity.WARNING,
        timeout=5.0,
    )
    argv = shlex.split(torchtyc_command(config))
    child = _config_from(build_parser().parse_args(argv[argv.index("lsp") :]), str(tmp_path))
    assert child.python == config.python
    assert child.variadic_rank == 4
    assert child.severity is Severity.WARNING
    assert child.timeout == 5.0
    assert "unused-dim" in child.ignore
