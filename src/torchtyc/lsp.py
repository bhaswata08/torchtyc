"""The language server.

Two things shape the design:

  * The lint pass is cheap and the trace pass is not. Typing publishes lint
    diagnostics immediately and schedules the trace behind a debounce, so the
    editor never waits on a torch import.
  * Tracing imports the file being edited, which means running module-level
    code. Doing that on every keystroke would be hostile, so a trace happens on
    open, on save, and after the buffer has been quiet for a moment.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from lsprotocol import types as lsp

from . import __version__
from . import config as config_module
from .config import Config, Overrides
from .diagnostics import Diagnostic, Severity
from .discovery import Target, scan_source
from .engine import Report, apply_suppressions, check_paths, lint_scan

try:  # pygls 2.x
    from pygls.lsp.server import LanguageServer
except ImportError:  # pygls 1.x
    from pygls.server import LanguageServer  # type: ignore[no-redef]

log = logging.getLogger("torchtyc")

DEBOUNCE_SECONDS = 0.7

_SEVERITY = {
    Severity.ERROR: lsp.DiagnosticSeverity.Error,
    Severity.WARNING: lsp.DiagnosticSeverity.Warning,
    Severity.INFO: lsp.DiagnosticSeverity.Information,
}


def uri_to_path(uri: str) -> str:
    return unquote(urlparse(uri).path)


def path_to_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def to_lsp(diagnostic: Diagnostic) -> lsp.Diagnostic:
    message = diagnostic.message
    if diagnostic.expected is not None or diagnostic.got is not None:
        message += f"\n  expected: {diagnostic.expected}\n  got:      {diagnostic.got}"
    if diagnostic.hint:
        message += f"\n  hint: {diagnostic.hint}"
    if diagnostic.note:
        message += f"\n  note: {diagnostic.note}"
    if diagnostic.suggestion:
        message += f"\n  try: {diagnostic.suggestion}"

    end_line = diagnostic.end_line if diagnostic.end_line is not None else diagnostic.line
    end_column = (
        diagnostic.end_column if diagnostic.end_column is not None else diagnostic.column + 1
    )

    return lsp.Diagnostic(
        range=lsp.Range(
            start=lsp.Position(line=diagnostic.line, character=diagnostic.column),
            end=lsp.Position(line=end_line, character=end_column),
        ),
        severity=_SEVERITY[diagnostic.severity],
        code=diagnostic.rule,
        code_description=lsp.CodeDescription(
            href=f"https://github.com/bhaswata08/torchtyc#{diagnostic.rule}"
        ),
        source="torchtyc",
        message=message,
    )


class TorchtycServer(LanguageServer):
    def __init__(self) -> None:
        super().__init__(name="torchtyc", version=__version__)
        self.overrides: Overrides = Overrides()
        self.pending: dict[str, asyncio.Task] = {}
        # One trace at a time per file. Cancelling the task that awaits a trace
        # does not stop the worker subprocess already running in its thread, so
        # without this a fast typist stacks up torch-importing processes that
        # each run to completion or to the timeout.
        self.tracing: dict[str, asyncio.Lock] = {}
        self.wanted: dict[str, int] = {}
        self.reports: dict[str, Report] = {}
        self.scans: dict[str, Any] = {}

    def source_of(self, uri: str) -> str:
        return self.workspace.get_text_document(uri).source

    def config_for(self, path: str) -> Config:
        """Reload config per file, so a monorepo with several projects works.

        What the user typed on the command line is applied on top, so it wins
        wherever the file happens to live.
        """
        return self.overrides.apply(config_module.load(path))

    def publish(self, uri: str, diagnostics: list[Diagnostic]) -> None:
        self.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=[to_lsp(d) for d in diagnostics])
        )

    def lint_now(self, uri: str) -> None:
        """The fast pass: parse only, no import, safe on every keystroke."""
        path = uri_to_path(uri)
        source = self.source_of(uri)
        config = self.config_for(path)
        scan = scan_source(source, path)
        self.scans[uri] = scan
        diagnostics = [
            d
            for d in apply_suppressions(lint_scan(scan, config), scan.suppressions)
            if d.rule not in config.ignore and d.severity <= config.severity
        ]
        # Keep whatever the last trace found, so results do not flicker away
        # while the user types.
        previous = self.reports.get(uri)
        if previous is not None:
            traced = [d for d in previous.diagnostics if d.rule in _TRACE_RULES]
            diagnostics = diagnostics + traced
        self.publish(uri, diagnostics)

    async def trace_soon(self, uri: str, delay: float = DEBOUNCE_SECONDS) -> None:
        existing = self.pending.pop(uri, None)
        if existing is not None:
            existing.cancel()

        async def run() -> None:
            try:
                if delay:
                    await asyncio.sleep(delay)
                await self.trace_now(uri)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("trace failed for %s", uri)

        self.pending[uri] = asyncio.create_task(run())

    async def trace_now(self, uri: str) -> None:
        generation = self.wanted.get(uri, 0) + 1
        self.wanted[uri] = generation
        lock = self.tracing.setdefault(uri, asyncio.Lock())
        async with lock:
            if self.wanted.get(uri) != generation:
                # A newer trace was asked for while this one queued. Its result
                # would be the one published anyway, so skip the subprocess.
                return
            await self._trace_once(uri)

    async def _trace_once(self, uri: str) -> None:
        path = uri_to_path(uri)
        source = self.source_of(uri)
        config = self.config_for(path)

        report = await asyncio.to_thread(check_paths, [path], config, {path: source}, True)
        self.reports[uri] = report

        diagnostics = list(report.diagnostics)
        if report.worker_error:
            diagnostics.append(
                Diagnostic(
                    path=path,
                    line=0,
                    column=0,
                    rule="trace-error",
                    severity=Severity.WARNING,
                    message=f"torchtyc could not trace this file: {report.worker_error}",
                    hint=f"interpreter: {config.interpreter}",
                )
            )
        self.publish(uri, diagnostics)


_TRACE_RULES = frozenset(
    {
        "shape-mismatch",
        "rank-mismatch",
        "dtype-mismatch",
        "dim-inconsistent",
        "not-a-tensor",
        "tuple-arity",
        "device-mismatch",
        "attribute-mismatch",
        "trace-error",
        "import-error",
        "uninstantiable",
        "unresolved-arg",
        "local-definition",
        "suppression-unused",
    }
)


server = TorchtycServer()


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: TorchtycServer, params: lsp.DidOpenTextDocumentParams) -> None:
    ls.lint_now(params.text_document.uri)
    await ls.trace_soon(params.text_document.uri, delay=0.0)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
async def did_change(ls: TorchtycServer, params: lsp.DidChangeTextDocumentParams) -> None:
    ls.lint_now(params.text_document.uri)
    await ls.trace_soon(params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
async def did_save(ls: TorchtycServer, params: lsp.DidSaveTextDocumentParams) -> None:
    await ls.trace_soon(params.text_document.uri, delay=0.0)


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: TorchtycServer, params: lsp.DidCloseTextDocumentParams) -> None:
    uri = params.text_document.uri
    task = ls.pending.pop(uri, None)
    if task is not None:
        task.cancel()
    ls.reports.pop(uri, None)
    ls.scans.pop(uri, None)
    ls.publish(uri, [])


def _target_at(ls: TorchtycServer, uri: str, line: int) -> Target | None:
    scan = ls.scans.get(uri)
    if scan is None:
        return None
    best: Target | None = None
    for target in scan.targets:
        # The cursor has to be inside the function, or module-level code below
        # the last one would show that function's shapes.
        if target.position.line <= line <= target.end_line and (
            best is None or target.position.line > best.position.line
        ):
            best = target
    return best


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: TorchtycServer, params: lsp.HoverParams) -> lsp.Hover | None:
    uri = params.text_document.uri
    report = ls.reports.get(uri)
    target = _target_at(ls, uri, params.position.line)
    if report is None or target is None:
        return None

    shapes = report.shapes_in(uri_to_path(uri)).get(target.qualname)
    if not shapes:
        return None

    lines = [f"**{target.qualname}** traced shapes", "", "```"]
    width = max(len(name) for name in shapes)
    for name, shape in shapes.items():
        label = "return" if name == "return" else name
        lines.append(f"{label:<{width}} {'->' if name == 'return' else ' :'} {shape}")
    lines.append("```")
    lines.append("")
    lines.append("_Each dimension name is bound to a distinct prime from 101._")

    return lsp.Hover(
        contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value="\n".join(lines))
    )


@server.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hints(ls: TorchtycServer, params: lsp.InlayHintParams) -> list[lsp.InlayHint]:
    uri = params.text_document.uri
    report = ls.reports.get(uri)
    scan = ls.scans.get(uri)
    if report is None or scan is None:
        return []

    traced = report.shapes_in(uri_to_path(uri))
    document = ls.workspace.get_text_document(uri)
    hints: list[lsp.InlayHint] = []
    for target in scan.targets:
        shapes = traced.get(target.qualname)
        if not shapes:
            continue
        # After the signature, not after the function name. Anchoring at
        # `position.end_column` put the hint between `def forward` and its
        # parameter list, splitting the very line it annotates.
        line = target.signature_end_line
        if not (params.range.start.line <= line <= params.range.end.line):
            continue
        summary = shapes.get("return")
        if summary is None:
            continue
        try:
            character = len(document.lines[line].rstrip("\r\n"))
        except IndexError:
            continue
        hints.append(
            lsp.InlayHint(
                position=lsp.Position(line=line, character=character),
                label=f"  traced -> {summary}",
                kind=lsp.InlayHintKind.Type,
                padding_left=True,
                tooltip=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value="\n".join(f"`{k}`: {v}" for k, v in shapes.items()),
                ),
            )
        )
    return hints


@server.feature(lsp.TEXT_DOCUMENT_CODE_LENS)
def code_lens(ls: TorchtycServer, params: lsp.CodeLensParams) -> list[lsp.CodeLens]:
    uri = params.text_document.uri
    scan = ls.scans.get(uri)
    report = ls.reports.get(uri)
    if scan is None:
        return []

    traced = report.shapes_in(uri_to_path(uri)) if report else {}
    lenses: list[lsp.CodeLens] = []
    for target in scan.targets:
        if not target.has_array_annotation:
            continue
        shapes = traced.get(target.qualname) or {}
        failing = [
            d
            for d in (report.diagnostics if report else [])
            if d.function == target.qualname and d.severity is Severity.ERROR
        ]
        if failing:
            title = f"$(error) {len(failing)} shape error(s)"
        elif shapes:
            title = f"traced: {shapes.get('return', 'ok')}"
        else:
            title = "not traced"
        lenses.append(
            lsp.CodeLens(
                range=lsp.Range(
                    start=lsp.Position(line=target.position.line, character=0),
                    end=lsp.Position(line=target.position.line, character=0),
                ),
                command=lsp.Command(
                    title=title,
                    command="torchtyc.trace",
                    arguments=[uri, target.qualname],
                ),
            )
        )
    return lenses


@server.feature(lsp.TEXT_DOCUMENT_CODE_ACTION)
def code_action(ls: TorchtycServer, params: lsp.CodeActionParams) -> list[lsp.CodeAction]:
    """Offer to silence a diagnostic, or to adopt the shape that was traced."""
    uri = params.text_document.uri
    document = ls.workspace.get_text_document(uri)
    actions: list[lsp.CodeAction] = []

    for diagnostic in params.context.diagnostics:
        if diagnostic.source != "torchtyc" or not isinstance(diagnostic.code, str):
            continue
        rule = diagnostic.code
        line = diagnostic.range.start.line
        try:
            text = document.lines[line].rstrip("\r\n")
        except IndexError:
            continue

        actions.append(
            lsp.CodeAction(
                title=f"Silence torchtyc[{rule}] on this line",
                kind=lsp.CodeActionKind.QuickFix,
                diagnostics=[diagnostic],
                edit=_replace_line(uri, line, f"{text}  # torchtyc: ignore[{rule}]"),
            )
        )

        suggestion = _extract(diagnostic.message, "try:")
        edit = (
            _rewrite_annotation(uri, document, diagnostic.range, suggestion) if suggestion else None
        )
        if edit is not None:
            actions.append(
                lsp.CodeAction(
                    title=f"Change the annotation to {suggestion}",
                    kind=lsp.CodeActionKind.QuickFix,
                    diagnostics=[diagnostic],
                    edit=edit,
                    is_preferred=True,
                )
            )

    return actions


def _extract(message: str, marker: str) -> str | None:
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            return stripped[len(marker) :].strip()
    return None


def _replace_line(uri: str, line: int, text: str) -> lsp.WorkspaceEdit:
    return lsp.WorkspaceEdit(
        changes={
            uri: [
                lsp.TextEdit(
                    range=lsp.Range(
                        start=lsp.Position(line=line, character=0),
                        end=lsp.Position(line=line, character=len(text) + 200),
                    ),
                    new_text=text,
                )
            ]
        }
    )


# A jaxtyping annotation: a dtype class, then a bracketed array type and dim
# string. The dim string is quoted, which is what tells it apart from an
# ordinary subscript such as `weights[0]`.
_ANNOTATION = re.compile(r'[A-Za-z_]\w*\s*\[[^][]*"[^"]*"\s*\]')


def _rewrite_annotation(
    uri: str, document: Any, span: lsp.Range, suggestion: str
) -> lsp.WorkspaceEdit | None:
    """Replace exactly the annotation the diagnostic points at.

    The range comes from the diagnostic rather than from searching the line,
    because searching cannot tell which annotation is meant. A tuple return
    holds several, and the suggestion describes one element while the
    diagnostic anchors the whole tuple: picking the last match there would
    rewrite the wrong element.

    The whole annotation is replaced, not only its dim string, since a dtype
    mismatch suggests the same dims under a different dtype class and
    rewriting the dims alone would change nothing.

    Nothing is offered unless the range holds exactly one array annotation, so
    a suggestion that does not describe the anchored text is declined rather
    than applied to the wrong place.
    """
    if span.start.line != span.end.line:
        return None
    try:
        text = document.lines[span.start.line].rstrip("\n")
    except IndexError:
        return None

    target = text[span.start.character : span.end.character]
    if _ANNOTATION.fullmatch(target) is None or target == suggestion:
        return None

    return lsp.WorkspaceEdit(changes={uri: [lsp.TextEdit(range=span, new_text=suggestion)]})


@server.feature(lsp.WORKSPACE_DID_CHANGE_CONFIGURATION)
async def did_change_configuration(ls: TorchtycServer, params: Any) -> None:
    # Every path reloads the config from disk, so there is nothing to refresh
    # here beyond rechecking the files already open.
    for uri in list(ls.reports):
        await ls.trace_soon(uri, delay=0.0)


@server.command("torchtyc.trace")
async def command_trace(ls: TorchtycServer, args: list[Any]) -> dict[str, Any]:
    uri = args[0]
    qualname = args[1] if len(args) > 1 else None
    await ls.trace_now(uri)
    report = ls.reports.get(uri)
    if report is None:
        return {}
    traced = report.shapes_in(uri_to_path(uri))
    return traced.get(qualname, {}) if qualname else traced


@server.command("torchtyc.recheck")
async def command_recheck(ls: TorchtycServer, args: list[Any]) -> None:
    for uri in list(ls.reports) or ([args[0]] if args else []):
        await ls.trace_now(uri)


def serve(tcp_port: int | None = None, overrides: Overrides | None = None) -> int:
    server.overrides = overrides or Overrides()
    if tcp_port:
        server.start_tcp("127.0.0.1", tcp_port)
    else:
        server.start_io()
    return 0
