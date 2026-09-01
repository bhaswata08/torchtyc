"""The subprocess that actually imports user code.

Run as `python -m torchtyc.worker`, reading one JSON job on stdin and writing
one JSON result on stdout. It lives in a separate process for three reasons:

  * it must run under the *project's* interpreter, with the project's torch,
    which is rarely the interpreter hosting the language server
  * importing user code runs module-level side effects, and a fresh process is
    the only honest way to pick up an edit without stale-module games
  * user code that segfaults or hangs takes the worker down, not the editor
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import linecache
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from .binding import DimBinder
from .diagnostics import RULES, Diagnostic, Severity
from .discovery import Attribute, ClassInfo, Position, Target, scan_source
from .tracing import (
    NotLive,
    TraceFailed,
    TraceResult,
    TraceSkipped,
    check_return,
    describe,
    instantiate,
    live_init,
    resolve_qualname,
    trace,
)


def import_from_path(path: Path, source: str | None = None) -> Any:
    """Import a file as part of its package, so relative imports resolve.

    With `source` given, that text is executed instead of the file on disk. The
    module keeps the dotted name and the `__file__` it would have had, so
    relative imports still resolve and a traceback still points at the real
    path. That is what lets the editor check a buffer before it is saved.
    """
    path = path.resolve()
    root = path.parent
    # `pkg/__init__.py` is the module `pkg`, not `pkg.__init__`: importing the
    # latter runs the package body a second time under a second name.
    parts = [] if path.stem == "__init__" else [path.stem]
    while (root / "__init__.py").exists():
        parts.append(root.name)
        root = root.parent

    dotted = ".".join(reversed(parts))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Drop any cached copy so the current text is what gets checked.
    for name in [n for n in sys.modules if n == dotted or n.startswith(dotted + ".")]:
        del sys.modules[name]

    if source is None:
        return importlib.import_module(dotted)

    package, _, _ = dotted.rpartition(".")
    if package:
        importlib.import_module(package)

    spec = importlib.util.spec_from_file_location(dotted, str(path))
    if spec is None:
        raise ImportError(f"cannot build a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[dotted] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    except BaseException:
        sys.modules.pop(dotted, None)
        raise
    return module


def _format_traceback(exc: BaseException, binder: DimBinder) -> str:
    """The traceback, with axis names put back into the exception message.

    Renaming the whole blob would be wrong: a frame line carries `line 101`,
    which is a source position and not a shape. `format_exception` indents every
    frame block by two spaces and leaves the message lines flush, so the split
    is exact and only the message ever passes through the binder.
    """
    parts = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(part if part.startswith("  ") else binder.rename_primes(part) for part in parts)


def _anchor(
    exc: BaseException,
    path: str,
    fallback: Position,
    binder: DimBinder,
    others: list[tuple[int, int, str]] | None = None,
) -> tuple[Position, str, str | None]:
    """Point at the deepest frame inside the file that torchtyc is not already checking.

    A shape error usually surfaces several frames down, inside einsum or matmul.
    The line the user needs to see is the last one they wrote, not torch's, so
    the deepest frame in the file normally wins. An unannotated helper is part
    of the caller's own code that way, which is what the reader wants.

    A module that calls another annotated module in the same file is different.
    That callee is checked in its own right and its code is shared by every
    caller, so its failing line says nothing about which caller passed the wrong
    shape. `others` carries the body spans of the file's other traced targets,
    and frames inside them are stepped over until a line the checked function
    actually owns is reached.

    The span comes from the frame's column offsets rather than its source text,
    because `FrameSummary.line` is stripped of leading indentation: measuring it
    would start the underline inside the indent and stop short of the statement.
    """
    summaries = traceback.extract_tb(exc.__traceback__)
    here = Path(path).resolve()
    mine = [
        (summary, frame)
        for summary, frame in zip(summaries, _live_frames(exc))
        if summary.filename and Path(summary.filename).resolve() == here
    ]
    spans = others or []
    outside = [pair for pair in mine if not _within(pair[0].lineno, spans)]
    picked = (outside or mine)[-1] if mine else None
    text = _format_traceback(exc, binder)
    hint = _hint(mine, picked, spans, binder)
    if picked is None:
        return fallback, text, hint
    chosen = picked[0]
    line = chosen.lineno - 1 if chosen.lineno else fallback.line
    end_line = chosen.end_lineno - 1 if chosen.end_lineno else line
    if chosen.colno is not None and chosen.end_colno is not None:
        return Position(line, chosen.colno, end_line, chosen.end_colno), text, hint
    # No column information: fall back to underlining the whole line.
    return Position(line, 0, line, len(chosen.line or "") or 1), text, hint


def _live_frames(exc: BaseException) -> list[Any]:
    """The frame objects behind a traceback, still holding their locals."""
    frames: list[Any] = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    return frames


def _hint(
    mine: list[tuple[traceback.FrameSummary, Any]],
    picked: tuple[traceback.FrameSummary, Any] | None,
    spans: list[tuple[int, int, str]],
    binder: DimBinder,
) -> str | None:
    """What to say under a trace error, beyond torch's own words.

    Two things the reader cannot see from the underlined line: where the error
    was actually raised, when that is somewhere further down, and what shapes
    the tensors on that line were carrying.
    """
    if not mine:
        return None
    deepest, frame = mine[-1]
    shapes = _shapes_on_line(frame, _statement(deepest), binder)
    if picked is None or picked is mine[-1] or not deepest.lineno:
        return shapes or None
    owner = _owner_of(deepest.lineno, spans) or deepest.name
    where = f"raised further down, in `{owner}` at line {deepest.lineno}"
    return f"{where}, where {shapes}" if shapes else where


_DERIVED = re.compile(r"<from ([^>]+)>")


def _derived_note(*texts: str | None) -> str | None:
    """Explain a `<from ...>` width, when one reached the reader.

    The marker stands where a number used to, and a number there was worse than
    useless: it was computed from the width torchtyc traced with, so it named a
    size the model does not have at any real width. What the width follows is
    the part that holds, and this says so once, in one place.
    """
    shown = sorted({match.group(0) for text in texts if text for match in _DERIVED.finditer(text)})
    if not shown:
        return None
    if len(shown) == 1:
        dims = _DERIVED.match(shown[0]).group(1)  # type: ignore[union-attr]
        return f"{shown[0]} is a width your __init__ computed from {dims}, so it is not {dims}"
    return "a <from ...> width is one your __init__ computed, so it is not that dimension itself"


# A name, or a dotted path such as `self.weight`, as it appears in source.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_STRING = re.compile(r"(\"\"\"|\'\'\'|\"|\').*?\1", re.DOTALL)
_MAX_SHAPES = 4


def _statement(summary: traceback.FrameSummary) -> str:
    """The whole statement that raised, not just its first line.

    An einsum call is usually written across several lines, and the tensors go
    on the ones after the first. `FrameSummary.line` carries only the line at
    `lineno`, so the source is read again across the statement's own span.
    """
    end = summary.end_lineno or summary.lineno
    if not summary.filename or not summary.lineno:
        return summary.line or ""
    lines = linecache.getlines(summary.filename)[summary.lineno - 1 : end]
    return "".join(lines) if lines else (summary.line or "")


def _shapes_on_line(frame: Any, text: str | None, binder: DimBinder) -> str:
    """The shapes of the tensors named on the line that raised, in the user's axis names.

    torch describes a shape bug in its own terms: an operand index and a
    subscript letter. Neither is anything the reader wrote. The names on that
    line are, so every one of them holding a tensor is rendered beside it, and
    a size that came out of the model's construction rather than an annotation
    shows up as the plain number it is.
    """
    found: dict[str, str] = {}
    # An einops pattern is a string of axis names. They are not locals, and a
    # local that happens to share one of their names is not what the line reads.
    for match in _NAME.finditer(_STRING.sub('""', text or "")):
        name = match.group(0)
        if name in found:
            continue
        value = _value_of(frame, name)
        if isinstance(value, torch.Tensor):
            found[name] = binder.render_shape(tuple(value.shape))
        if len(found) == _MAX_SHAPES:
            break
    return ", ".join(f"{name} is {shape}" for name, shape in found.items())


def _value_of(frame: Any, dotted: str) -> Any:
    """What a name on the failing line refers to, or None if it refers to nothing.

    Attribute reads can run user code through a property, so a raising getattr
    means the name is simply not reportable.
    """
    root, *rest = dotted.split(".")
    if root in frame.f_locals:
        value = frame.f_locals[root]
    elif root in frame.f_globals:
        value = frame.f_globals[root]
    else:
        return None
    for attr in rest:
        try:
            value = getattr(value, attr)
        except Exception:  # noqa: BLE001
            return None
    return value


def _owner_of(lineno: int | None, spans: list[tuple[int, int, str]]) -> str | None:
    """The qualname of the target whose body holds a 1-based traceback line."""
    if lineno is None:
        return None
    return next((name for first, last, name in spans if first <= lineno - 1 <= last), None)


def _within(lineno: int | None, spans: list[tuple[int, int, str]]) -> bool:
    return _owner_of(lineno, spans) is not None


def _body_spans(targets: list[Target], skip: Target) -> list[tuple[int, int, str]]:
    """Body line ranges of every traced target in the file except the one being checked."""
    return [
        (t.signature_end_line, t.end_line, t.qualname)
        for t in targets
        if t is not skip and t.end_line > 0 and t.end_line >= t.signature_end_line
    ]


def _severity(rule: str) -> Severity:
    entry = RULES.get(rule)
    return entry.severity if entry else Severity.ERROR


def check_target(
    module: Any,
    target: Target,
    path: str,
    variadic_rank: int,
    siblings: list[Target] | None = None,
) -> tuple[list[Diagnostic], TraceResult | None]:
    """Trace one target once, returning its diagnostics and the trace itself.

    The trace is handed back so hover can be derived from it: tracing runs the
    forward pass, and the editor asks for both on every keystroke.
    """
    out: list[Diagnostic] = []
    anchor = target.returns_position or target.position

    try:
        result = trace(module, target, variadic_rank)
    except NotLive:
        return [], None
    except TraceFailed as exc:
        position, text, hint = _anchor(
            exc.error, path, target.position, exc.binder, _body_spans(siblings or [], target)
        )
        message = exc.binder.rename_primes(f"{type(exc.error).__name__}: {exc.error}")
        return [
            Diagnostic(
                path=path,
                line=position.line,
                column=position.column,
                end_line=position.end_line,
                end_column=position.end_column,
                rule="trace-error",
                severity=Severity.ERROR,
                message=message,
                function=target.qualname,
                hint=hint,
                note=_derived_note(message, hint),
                traceback=text,
            )
        ], None
    except TraceSkipped as exc:
        return [
            Diagnostic(
                path=path,
                line=target.position.line,
                column=target.position.column,
                end_line=target.position.end_line,
                end_column=target.position.end_column,
                rule=exc.rule,
                severity=_severity(exc.rule),
                message=exc.message,
                function=target.qualname,
                hint=exc.hint or None,
            )
        ], None

    for problem in check_return(target.returns, result.returned, result.binder):
        out.append(
            Diagnostic(
                path=path,
                line=anchor.line,
                column=anchor.column,
                end_line=anchor.end_line,
                end_column=anchor.end_column,
                rule=problem["rule"],
                severity=_severity(problem["rule"]),
                message=f"in the return of `{target.qualname}`: {problem['message']}",
                function=target.qualname,
                expected=problem.get("expected"),
                got=problem.get("got"),
                hint=problem.get("hint") or None,
                suggestion=problem.get("suggestion"),
            )
        )

    return out, result


def check_attributes(
    module: Any, info: ClassInfo, path: str, variadic_rank: int
) -> list[Diagnostic]:
    """Construct the class once and compare `self.X` against its annotation."""
    binder = DimBinder(variadic_rank=variadic_rank)
    attributes: list[Attribute] = []

    try:
        cls = resolve_qualname(
            module,
            info.qualname,
            conditional=info.conditional,
            span=(info.def_line, info.end_line),
        )
        # The annotations to check are the ones the live constructor wrote, so
        # a guarded `__init__` that this import skipped reports nothing.
        chosen = live_init(info, cls, module)
        attributes = chosen.attributes if chosen is not None else []
        instance = instantiate(info, cls, binder, info.dim_names, module)
    except NotLive:
        return []
    except TraceSkipped as exc:
        return [
            Diagnostic(
                path=path,
                line=info.position.line,
                column=info.position.column,
                end_line=info.position.end_line,
                end_column=info.position.end_column,
                rule=exc.rule,
                severity=_severity(exc.rule),
                message=exc.message,
                function=info.qualname,
                hint=exc.hint or None,
            )
        ]
    except Exception as exc:  # noqa: BLE001
        position, text, _ = _anchor(exc, path, info.position, binder)
        return [
            Diagnostic(
                path=path,
                line=position.line,
                column=position.column,
                end_line=position.end_line,
                end_column=position.end_column,
                rule="trace-error",
                severity=Severity.ERROR,
                message=binder.rename_primes(
                    f"constructing `{info.qualname}`: {type(exc).__name__}: {exc}"
                ),
                function=info.qualname,
                traceback=text,
            )
        ]

    out: list[Diagnostic] = []
    for attribute in attributes:
        if not hasattr(instance, attribute.name):
            continue
        value = getattr(instance, attribute.name)
        for problem in check_return(attribute.spec, value, binder):
            rule = (
                "attribute-mismatch"
                if problem["rule"] in ("shape-mismatch", "rank-mismatch")
                else problem["rule"]
            )
            out.append(
                Diagnostic(
                    path=path,
                    line=attribute.position.line,
                    column=attribute.position.column,
                    end_line=attribute.position.end_line,
                    end_column=attribute.position.end_column,
                    rule=rule,
                    severity=_severity(rule),
                    message=f"`self.{attribute.name}`: {problem['message']}",
                    function=info.qualname,
                    expected=problem.get("expected"),
                    got=problem.get("got"),
                    hint=problem.get("hint") or None,
                    suggestion=problem.get("suggestion"),
                )
            )
    return out


def shapes_for_hover(result: TraceResult) -> dict[str, str]:
    """Argument and return shapes of a completed trace, for hover and inlay hints."""
    hints = {
        name: result.binder.render_shape(shape) for name, shape in result.argument_shapes.items()
    }
    hints["return"] = describe(result.returned, result.binder)
    return hints


def _is_local(qualname: str) -> bool:
    return "<locals>" in qualname


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    variadic_rank = job.get("variadic_rank", 2)
    want_hover = job.get("hover", False)
    diagnostics: list[dict[str, Any]] = []
    # path -> qualname -> shapes. Two files in one job can define the same
    # qualname, so the path has to be part of the key.
    hovers: dict[str, dict[str, dict[str, str]]] = {}

    for path in job["paths"]:
        buffer = job.get("sources", {}).get(path)
        source = buffer if buffer is not None else Path(path).read_text(encoding="utf-8")
        scan = scan_source(source, path)
        if scan.syntax_error is not None:
            continue  # the in-process pass already reported it

        targets = [t for t in scan.targets if t.has_array_annotation]
        # A class inside a function body reports once, at the class line, so a
        # factory with several annotated methods does not repeat itself. A local
        # class with nothing annotated is not reported at all: there is no
        # coverage to miss, so saying so would be noise.
        annotated_owners = {t.owner.qualname for t in targets if t.owner is not None}
        local_classes = {
            c.qualname
            for c in scan.classes
            if _is_local(c.qualname) and (c.all_attributes or c.qualname in annotated_owners)
        }
        targets = [t for t in targets if t.owner is None or t.owner.qualname not in local_classes]
        classes = [c for c in scan.classes if c.all_attributes or c.qualname in local_classes]
        if not targets and not classes:
            continue

        try:
            module = import_from_path(Path(path), buffer)
        except Exception as exc:  # noqa: BLE001
            position, text, _ = _anchor(exc, path, Position(0, 0, 0, 1), DimBinder())
            diagnostics.append(
                Diagnostic(
                    path=path,
                    line=position.line,
                    column=position.column,
                    end_line=position.end_line,
                    end_column=position.end_column,
                    rule="import-error",
                    severity=Severity.ERROR,
                    message=f"{type(exc).__name__}: {exc}",
                    traceback=text,
                ).to_json()
            )
            continue

        for info in classes:
            for diagnostic in check_attributes(module, info, path, variadic_rank):
                diagnostics.append(diagnostic.to_json())

        for target in targets:
            found, result = check_target(module, target, path, variadic_rank, targets)
            for diagnostic in found:
                diagnostics.append(diagnostic.to_json())
            if want_hover:
                hovers.setdefault(path, {})[target.qualname] = (
                    shapes_for_hover(result) if result is not None else {}
                )

    return {"diagnostics": diagnostics, "hovers": hovers}


def main() -> int:
    # stdout is this process's JSON channel, and importing user code runs
    # module-level prints. Hold the real stream and point `sys.stdout` at
    # stderr before any of that code can reach it.
    channel = sys.stdout
    sys.stdout = sys.stderr

    try:
        job = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        json.dump({"error": f"bad job: {exc}"}, channel)
        return 2

    try:
        result = run_job(job)
    except Exception as exc:  # noqa: BLE001
        result = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "diagnostics": [],
            "hovers": {},
        }
    json.dump(result, channel)
    channel.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
