"""One stdio pipe, several language servers behind it.

Editors that allow only one language server per filetype, or users who would
rather configure one, can point at `torchtyc mux` and get basedpyright and
torchtyc together. The mux speaks LSP to the editor and LSP to each downstream
server, and reconciles the two sides:

  * request ids are rewritten per server, because two servers will both answer
    with id 1 and the editor must see exactly one answer
  * results are merged per method: lists concatenate, hovers stack, and
    anything else takes the first server that actually answered
  * diagnostics are held per server and republished as a union, so one server
    clearing its list does not wipe the other's findings

Servers only ever see ids they issued, so a server that requests something from
the client (configuration, dynamic registration) still works.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any

from .config import Config

log = logging.getLogger("torchtyc.mux")

# Methods whose results are lists and can simply be concatenated.
_CONCAT = {
    "textDocument/codeAction",
    "textDocument/codeLens",
    "textDocument/inlayHint",
    "textDocument/documentSymbol",
    "textDocument/documentHighlight",
    "textDocument/references",
    "textDocument/definition",
    "textDocument/typeDefinition",
    "textDocument/implementation",
    "textDocument/declaration",
    "textDocument/semanticTokens/full/delta",
    "workspace/symbol",
}


async def _read_message(stream: asyncio.StreamReader) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = await stream.readline()
        if not line:
            return None
        text = line.decode("ascii", "replace").strip()
        if not text:
            break
        if ":" in text:
            key, value = text.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    length = int(headers.get("content-length", 0))
    if length <= 0:
        return None
    body = await stream.readexactly(length)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        log.warning("dropping unparseable message")
        return None


def _encode(message: dict[str, Any]) -> bytes:
    body = json.dumps(message).encode("utf-8")
    return b"Content-Length: %d\r\n\r\n" % len(body) + body


@dataclass
class Downstream:
    name: str
    process: asyncio.subprocess.Process
    # Diagnostics this server last published, keyed by document uri.
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(_encode(message))


@dataclass
class Pending:
    """A client request fanned out to several servers."""

    client_id: Any
    method: str
    expected: int
    results: list[Any] = field(default_factory=list)
    errors: list[Any] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.results) + len(self.errors) >= self.expected


class Mux:
    def __init__(self, commands: list[str]) -> None:
        self.commands = commands
        self.servers: list[Downstream] = []
        self.pending: dict[int, Pending] = {}
        # (server index, id it issued) for a request the server sent upstream.
        self.upstream: dict[int, tuple[int, Any]] = {}
        # id the mux issued downstream -> the pending fan-out it belongs to.
        self.downstream: dict[tuple[int, Any], int] = {}
        self.counter = 0
        self.out_lock = asyncio.Lock()
        self.writer: asyncio.StreamWriter | None = None

    def next_id(self) -> int:
        self.counter += 1
        return self.counter

    async def start(self) -> None:
        for command in self.commands:
            argv = shlex.split(command)
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except OSError as exc:
                print(f"torchtyc mux: cannot start {argv[0]}: {exc}", file=sys.stderr)
                continue
            self.servers.append(Downstream(name=argv[0], process=process))

        if not self.servers:
            raise SystemExit("torchtyc mux: no downstream server started")

    async def to_client(self, message: dict[str, Any]) -> None:
        assert self.writer is not None
        async with self.out_lock:
            self.writer.write(_encode(message))
            await self.writer.drain()

    # ------------------------------------------------------------------ client

    async def pump_client(self, reader: asyncio.StreamReader) -> None:
        while True:
            message = await _read_message(reader)
            if message is None:
                break
            await self.from_client(message)

    async def from_client(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            await self.fan_out(message)
        elif "method" in message:
            for server in self.servers:
                server.send(message)
        else:
            # A response to something a server asked the client.
            entry = self.upstream.pop(message.get("id"), None)
            if entry is None:
                return
            index, original = entry
            self.servers[index].send({**message, "id": original})

    async def fan_out(self, message: dict[str, Any]) -> None:
        method = message["method"]
        pending = Pending(client_id=message["id"], method=method, expected=len(self.servers))
        token = self.next_id()
        self.pending[token] = pending

        for index, server in enumerate(self.servers):
            issued = self.next_id()
            self.downstream[(index, issued)] = token
            server.send({**message, "id": issued})

    # ------------------------------------------------------------------ server

    async def pump_server(self, index: int) -> None:
        server = self.servers[index]
        assert server.process.stdout is not None
        while True:
            message = await _read_message(server.process.stdout)
            if message is None:
                break
            await self.from_server(index, message)

    async def from_server(self, index: int, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            issued = self.next_id()
            self.upstream[issued] = (index, message["id"])
            await self.to_client({**message, "id": issued})
            return

        if "method" in message:
            if message["method"] == "textDocument/publishDiagnostics":
                await self.merge_diagnostics(index, message["params"])
            else:
                await self.to_client(message)
            return

        token = self.downstream.pop((index, message.get("id")), None)
        if token is None:
            return
        pending = self.pending.get(token)
        if pending is None:
            return

        if "error" in message:
            pending.errors.append(message["error"])
        else:
            result = message.get("result")
            pending.results.append(result)
            if pending.method == "initialize" and isinstance(result, dict):
                self.servers[index].capabilities = result.get("capabilities", {}) or {}

        if pending.complete:
            self.pending.pop(token, None)
            await self.reply(pending)

    async def reply(self, pending: Pending) -> None:
        if not pending.results and pending.errors:
            await self.to_client(
                {"jsonrpc": "2.0", "id": pending.client_id, "error": pending.errors[0]}
            )
            return

        await self.to_client(
            {"jsonrpc": "2.0", "id": pending.client_id, "result": self.merge(pending)}
        )

    def merge(self, pending: Pending) -> Any:
        results = [r for r in pending.results if r is not None]
        if not results:
            return None

        if pending.method == "initialize":
            merged: dict[str, Any] = {}
            for result in results:
                _deep_merge(merged, result)
            # The mux serves whatever any server serves, so advertise the union
            # and let the fan-out decide who answers.
            return merged

        if pending.method in _CONCAT:
            out: list[Any] = []
            for result in results:
                if isinstance(result, list):
                    out.extend(result)
                elif result is not None:
                    out.append(result)
            return out

        if pending.method == "textDocument/hover":
            parts: list[str] = []
            for result in results:
                contents = (result or {}).get("contents")
                text = _hover_text(contents)
                if text:
                    parts.append(text)
            if not parts:
                return None
            return {
                "contents": {"kind": "markdown", "value": "\n\n---\n\n".join(parts)},
            }

        return results[0]

    async def merge_diagnostics(self, index: int, params: dict[str, Any]) -> None:
        uri = params["uri"]
        self.servers[index].diagnostics[uri] = params.get("diagnostics", [])
        combined: list[Any] = []
        for server in self.servers:
            combined.extend(server.diagnostics.get(uri, []))
        await self.to_client(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {**params, "diagnostics": combined},
            }
        )


def _hover_text(contents: Any) -> str:
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return contents.get("value", "")
    if isinstance(contents, list):
        return "\n\n".join(_hover_text(item) for item in contents)
    return ""


def _deep_merge(into: dict[str, Any], other: dict[str, Any]) -> None:
    for key, value in other.items():
        if key not in into or into[key] is None:
            into[key] = value
        elif isinstance(into[key], dict) and isinstance(value, dict):
            _deep_merge(into[key], value)
        elif isinstance(into[key], list) and isinstance(value, list):
            into[key] = into[key] + [v for v in value if v not in into[key]]


async def _run(commands: list[str]) -> int:
    mux = Mux(commands)
    await mux.start()

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout.buffer
    )
    mux.writer = asyncio.StreamWriter(transport, protocol, None, loop)

    tasks = [asyncio.create_task(mux.pump_client(reader))]
    tasks += [asyncio.create_task(mux.pump_server(i)) for i in range(len(mux.servers))]

    _, remaining = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in remaining:
        task.cancel()
    for server in mux.servers:
        if server.process.returncode is None:
            server.process.terminate()
    return 0


def serve_mux(config: Config, servers: list[str]) -> int:
    """Run torchtyc's own server alongside the given commands."""
    commands = [f"{sys.executable} -m torchtyc.cli lsp", *servers]
    try:
        return asyncio.run(_run(commands))
    except KeyboardInterrupt:
        return 130
