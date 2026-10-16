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

import ast
import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import json
import linecache
import os
import re
import sys
import threading
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .binding import DimBinder
from .diagnostics import RULES, Diagnostic, Severity
from .discovery import Attribute, ClassInfo, Position, Target, scan_source
from .effects import active_guard, attempt_cooperative_stop, unwrap_blocked
from .sitepaths import is_system_path

# torch and everything that touches it load on first use, not at import. The
# parent starts the per-file timeout clock when this process announces
# `ready`, and `ready` has to come out before a torch import that can run for
# seconds. Until `_ensure_initialized` has run these names are None, so call
# it at the top of anything that traces.
if TYPE_CHECKING:
    import torch

    from .tracing import (
        NotLive,
        TraceFailed,
        TraceResult,
        TraceSkipped,
        _written_integers,
        check_return,
        describe,
        explain_derived_sizes,
        instantiate_retrying,
        live_init,
        resolve_qualname,
        trace,
    )
else:
    torch = None
    NotLive = None
    TraceFailed = None
    TraceResult = None
    TraceSkipped = None
    check_return = None
    describe = None
    explain_derived_sizes = None
    instantiate_retrying = None
    live_init = None
    resolve_qualname = None
    trace = None
    _written_integers = None


def _ensure_initialized() -> None:
    global torch, NotLive, TraceFailed, TraceResult, TraceSkipped
    global \
        check_return, \
        describe, \
        explain_derived_sizes, \
        instantiate_retrying, \
        live_init, \
        resolve_qualname, \
        trace, \
        _written_integers
    if torch is not None:
        return
    import torch as _torch

    from . import tracing as _tracing

    torch = _torch
    NotLive = _tracing.NotLive
    TraceFailed = _tracing.TraceFailed
    TraceResult = _tracing.TraceResult
    TraceSkipped = _tracing.TraceSkipped
    check_return = _tracing.check_return
    describe = _tracing.describe
    explain_derived_sizes = _tracing.explain_derived_sizes
    instantiate_retrying = _tracing.instantiate_retrying
    live_init = _tracing.live_init
    resolve_qualname = _tracing.resolve_qualname
    trace = _tracing.trace
    _written_integers = _tracing._written_integers


_BASELINE_MODULES: frozenset[str] = frozenset(sys.modules)


def _should_keep_module(name: str) -> bool:
    if name in _BASELINE_MODULES:
        return True
    if name == "torch" or name.startswith("torch."):
        return True
    if name == "torchtyc" or name.startswith("torchtyc."):
        return True

    mod = sys.modules.get(name)
    if mod is None:
        return False

    if name in sys.builtin_module_names:
        return True

    file = getattr(mod, "__file__", None)
    if file is not None:
        return is_system_path(Path(file))

    paths = getattr(mod, "__path__", None)
    if paths is not None:
        # A namespace package's `__path__` recomputes itself on iteration by
        # looking its parent up in `sys.modules`, and raises `KeyError` once the
        # parent is gone. A parent that is gone was dropped, so the child goes too.
        try:
            resolved_paths = [Path(p).resolve() for p in paths]
            return bool(resolved_paths) and all(is_system_path(p) for p in resolved_paths)
        except (ValueError, OSError, KeyError):
            return False

    top = name.partition(".")[0]
    return top in sys.stdlib_module_names or top in sys.builtin_module_names


def _restore_modules_and_path(saved_path: list[str]) -> None:
    sys.path[:] = saved_path
    # Decide everything before dropping anything: a namespace package reads its
    # parent out of `sys.modules` to answer, and the parent comes first here.
    doomed = [name for name in list(sys.modules) if not _should_keep_module(name)]
    for name in doomed:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    sys.path_importer_cache.clear()


def _prime_linecache(path: Path | str, source: str) -> tuple[str, ...]:
    lines = [line + "\n" for line in source.splitlines()]
    keys: list[str] = [str(path)]
    try:
        resolved = str(Path(path).resolve())
        if resolved != str(path):
            keys.append(resolved)
    except (ValueError, OSError):
        resolved = str(path)
    entry = (len(source), None, lines, resolved)
    for key in keys:
        linecache.cache[key] = entry
    return tuple(keys)


def _clear_linecache(paths: Iterable[Path | str]) -> None:
    for p in paths:
        raw = str(p)
        linecache.cache.pop(raw, None)
        try:
            resolved = str(Path(p).resolve())
            linecache.cache.pop(resolved, None)
        except (ValueError, OSError):
            pass


def _module_name(path: Path) -> tuple[Path, str]:
    """The sys.path root and dotted name a file imports under, packages included."""
    path = path.resolve()
    root = path.parent
    # `pkg/__init__.py` is the module `pkg`, not `pkg.__init__`.
    parts = [] if path.stem == "__init__" else [path.stem]
    while (root / "__init__.py").exists():
        parts.append(root.name)
        root = root.parent
    return root, ".".join(reversed(parts))


def import_from_path(path: Path, source: str | None = None) -> Any:
    """Import a file as part of its package, so relative imports resolve.

    With `source` given, that text is executed instead of the file on disk. The
    module keeps the dotted name and the `__file__` it would have had, so
    relative imports still resolve and a traceback still points at the real
    path. That is what lets the editor check a buffer before it is saved.
    """
    root, dotted = _module_name(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Drop any cached copy so the current text is what gets checked.
    for name in [n for n in sys.modules if n == dotted or n.startswith(dotted + ".")]:
        del sys.modules[name]

    if source is None:
        return importlib.import_module(dotted)

    _prime_linecache(path, source)

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
) -> tuple[Position, str, str | None, str | None]:
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
    statement = _statement(mine[-1][0]) if mine else None
    if picked is None:
        return fallback, text, hint, statement
    chosen = picked[0]
    line = chosen.lineno - 1 if chosen.lineno else fallback.line
    end_line = chosen.end_lineno - 1 if chosen.end_lineno else line
    if chosen.colno is not None and chosen.end_colno is not None:
        return Position(line, chosen.colno, end_line, chosen.end_colno), text, hint, statement
    # No column information: fall back to underlining the whole line.
    return Position(line, 0, line, len(chosen.line or "") or 1), text, hint, statement


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


def _note(message: str | None, hint: str | None, statement: str | None = None) -> str | None:
    """A note explaining synthetic widths or primes when present."""
    derived = _derived_note(message, hint)
    if derived is not None:
        return derived
    if not hint or "traced as" not in hint:
        return None

    combined = f"{message or ''} {statement or ''}"
    divisor_match = re.search(r"(?:%|//)\s*([A-Za-z_][A-Za-z0-9_]*)", combined)
    if divisor_match and divisor_match.group(1) in hint:
        divisor = divisor_match.group(1)
        return (
            f"torchtyc picks these widths itself; give `{divisor}` a default so it can pick ones that divide, "
            "or add # torchtyc: ignore[trace-error]"
        )
    if "%" in combined or "//" in combined:
        return (
            "torchtyc picks these widths itself; give parameter defaults so it can pick ones that divide, "
            "or add # torchtyc: ignore[trace-error]"
        )
    return "torchtyc picks these widths itself; add # torchtyc: ignore[trace-error]"


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


def _describe_integer(value: int, binder: DimBinder, issued: dict[int, str]) -> str | None:
    """Describe an integer value when it stands for a bound dimension or product."""
    if value in issued:
        return str(value)
    if binder.is_flattened(value):
        product_name = binder.describe(value)
        return f"{value} ({product_name})"
    return None


def _failed_condition(statement: str | None) -> str | None:
    """Extract the condition expression from an assert statement."""
    if not statement:
        return None
    text = statement.strip()
    try:
        tree = ast.parse(text)
        if tree.body and isinstance(tree.body[0], ast.Assert):
            node = tree.body[0]
            cond = ast.get_source_segment(text, node.test)
            if cond:
                return " ".join(cond.split())
            return ast.unparse(node.test)
    except Exception:  # noqa: BLE001, S110
        pass
    match = re.match(r"^assert\s+(.+?)(?:,.*)?$", text, re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return None


def _format_exc(exc: BaseException, statement: str | None = None, prefix: str = "") -> str:
    """Format an exception into a diagnostic message that never ends in a bare 'Type: '.

    When an exception has an empty str() (such as an argument-less AssertionError or
    RuntimeError), fall back to naming the failed condition from the raising statement,
    or simply the exception type name without a trailing colon.
    """
    exc_type = type(exc).__name__
    err_str = str(exc).strip()
    if err_str:
        msg = f"{exc_type}: {err_str}"
    elif isinstance(exc, AssertionError):
        cond = _failed_condition(statement)
        if cond:
            msg = f"AssertionError: {cond}"
        elif statement and statement.strip():
            msg = f"AssertionError: {statement.strip()}"
        else:
            msg = "AssertionError: assertion failed"
    else:
        msg = exc_type

    if prefix:
        return f"{prefix}{msg}"
    return msg


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
    if found:
        return ", ".join(f"{name} is {shape}" for name, shape in found.items())

    # When the failing line has no tensors, report bound integer dimensions.
    int_found: dict[str, str] = {}
    issued = binder.issued()

    for match in _NAME.finditer(_STRING.sub('""', text or "")):
        name = match.group(0)
        if name in int_found:
            continue
        value = _value_of(frame, name)
        if isinstance(value, int) and not isinstance(value, bool):
            desc = _describe_integer(value, binder, issued)
            if desc is not None:
                int_found[name] = desc
        if len(int_found) == _MAX_SHAPES:
            break

    if not int_found and frame is not None and hasattr(frame, "f_locals"):
        for var_name, value in frame.f_locals.items():
            if var_name.startswith("_") or var_name in int_found:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                desc = _describe_integer(value, binder, issued)
                if desc is not None:
                    int_found[var_name] = desc
            if len(int_found) == _MAX_SHAPES:
                break

    if int_found:
        items = list(int_found.items())
        parts = [
            f"{name} traced as {desc}" if i == 0 else f"{name} as {desc}"
            for i, (name, desc) in enumerate(items)
        ]
        if len(parts) == 1:
            joined = parts[0]
        elif len(parts) == 2:
            joined = f"{parts[0]} and {parts[1]}"
        else:
            joined = f"{', '.join(parts[:-1])} and {parts[-1]}"
        if _failed_condition(text) is not None or (text and text.strip().startswith("assert")):
            return f"{joined}, so the assertion cannot hold"
        return joined

    return ""


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


def _blocked_diagnostic(
    exc: BaseException,
    path: str,
    fallback: Position,
    binder: DimBinder,
    rule: str,
    function: str | None = None,
    others: list[tuple[int, int, str]] | None = None,
) -> Diagnostic | None:
    blocked = unwrap_blocked(exc)
    if blocked is None:
        return None
    position, _, _, _ = _anchor(exc, path, fallback, binder, others)
    culprit_msg = str(blocked)
    if blocked.culprit is not None and Path(blocked.culprit[0]).resolve() != Path(path).resolve():
        culprit_file = Path(blocked.culprit[0]).name
        culprit_msg = f"{blocked} (in {culprit_file} at line {blocked.culprit[1]})"
    return Diagnostic(
        path=path,
        line=position.line,
        column=position.column,
        end_line=position.end_line,
        end_column=position.end_column,
        rule=rule,
        severity=_severity(rule),
        message=f"BlockedEffect: {culprit_msg}",
        function=function,
        traceback=None,
    )


class TargetOutcome(tuple):
    diagnostics: list[Diagnostic]
    result: TraceResult | None
    checked: bool
    skipped: bool

    def __new__(
        cls,
        diagnostics: list[Diagnostic],
        result: TraceResult | None,
        *,
        checked: bool = False,
        skipped: bool = False,
    ):
        inst = super().__new__(cls, (diagnostics, result))
        inst.diagnostics = diagnostics
        inst.result = result
        inst.checked = checked
        inst.skipped = skipped
        return inst


def check_target(
    module: Any,
    target: Target,
    path: str,
    variadic_rank: int,
    siblings: list[Target] | None = None,
    *,
    allow_effects: bool = False,
) -> tuple[list[Diagnostic], TraceResult | None]:
    """Trace one target once, returning its diagnostics and the trace itself.

    The trace is handed back so hover can be derived from it: tracing runs the
    forward pass, and the editor asks for both on every keystroke.
    """
    out: list[Diagnostic] = []
    _ensure_initialized()
    anchor = target.returns_position or target.position

    try:
        with active_guard(enabled=not allow_effects, phase="trace"):
            result = trace(module, target, variadic_rank)
    except NotLive:
        return TargetOutcome([], None, checked=False, skipped=False)
    except TraceFailed as exc:
        diag = _blocked_diagnostic(
            exc.error,
            path,
            target.position,
            exc.binder,
            rule="trace-error",
            function=target.qualname,
            others=_body_spans(siblings or [], target),
        )
        if diag is not None:
            return TargetOutcome([diag], None, checked=True, skipped=False)
        position, text, hint, statement = _anchor(
            exc.error, path, target.position, exc.binder, _body_spans(siblings or [], target)
        )
        message = exc.binder.rename_primes(_format_exc(exc.error, statement))
        found = [
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
                note=_note(message, hint, statement),
                traceback=text,
            )
        ]
        if exc.first_error is not None:
            # The error only shows at the rescaled widths, so without them a
            # reader tracing at the usual ones stops at the first error instead.
            found.append(
                _retried_diagnostic(
                    target, path, exc.first_error, exc.binder, dict(exc.binder.sizes), failed=True
                )
            )
        return TargetOutcome(found, None, checked=True, skipped=False)
    except TraceSkipped as exc:
        pos = exc.position or target.position
        return TargetOutcome(
            [
                Diagnostic(
                    path=path,
                    line=pos.line,
                    column=pos.column,
                    end_line=pos.end_line,
                    end_column=pos.end_column,
                    rule=exc.rule,
                    severity=_severity(exc.rule),
                    message=exc.message,
                    function=target.qualname,
                    hint=exc.hint or None,
                )
            ],
            None,
            checked=False,
            skipped=True,
        )

    if result.retried and result.first_error is not None:
        out.append(
            _retried_diagnostic(
                target,
                path,
                result.first_error,
                result.binder,
                result.rescaled_widths,
                failed=False,
            )
        )

    problems = check_return(target.returns, result.returned, result.binder)
    if problems and result.built is not None:
        # A mismatch against a width the model worked out for itself reads as a
        # bare number, and that number only exists at the width torchtyc traced
        # with. Working out what it follows costs a handful of constructions,
        # so it waits until there is something to report, and then the check
        # runs again to write the messages with the names in. Every name is
        # already bound by now, so the second run compares the same sizes and
        # reaches the same verdict.
        explain_derived_sizes(result.binder, result.built)
        problems = check_return(target.returns, result.returned, result.binder)

    for problem in problems:
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
                note=_derived_note(problem["message"], problem.get("got")),
                suggestion=problem.get("suggestion"),
            )
        )

    return TargetOutcome(out, result, checked=True, skipped=False)


def _retried_diagnostic(
    target: Target,
    path: str,
    first: BaseException,
    binder: DimBinder,
    widths: dict[str, int],
    *,
    failed: bool,
) -> Diagnostic:
    """The `trace-retried` note: the widths a retry ran at and why it was needed."""
    statement = None
    try:
        _, _, _, statement = _anchor(first, path, target.position, binder)
    except Exception:  # noqa: BLE001, S110
        pass
    first_reason = binder.rename_primes(_format_exc(first, statement)).splitlines()[0]
    widths_str = ", ".join(f"{k}={v}" for k, v in widths.items())
    if failed:
        message = f"`{target.qualname}` failed on rescaled widths"
        hint = "the error above shows only at these widths; the usual ones stop at the first error"
    else:
        message = f"`{target.qualname}` traced on rescaled widths"
        hint = "the annotation was not checked at the widths torchtyc normally uses"
    return Diagnostic(
        path=path,
        line=target.position.line,
        column=target.position.column,
        end_line=target.position.end_line,
        end_column=target.position.end_column,
        rule="trace-retried",
        severity=Severity.INFO,
        message=f"{message}\n  {widths_str} ({first_reason})",
        function=target.qualname,
        hint=hint,
    )


def _constructor_error(
    exc: BaseException, path: str, info: ClassInfo, binder: DimBinder
) -> list[Diagnostic]:
    """The one diagnostic for a class whose constructor raised."""
    diag = _blocked_diagnostic(
        exc,
        path,
        info.position,
        binder,
        rule="trace-error",
        function=info.qualname,
    )
    if diag is not None:
        return [diag]
    position, text, hint, statement = _anchor(exc, path, info.position, binder)
    message = binder.rename_primes(
        _format_exc(exc, statement, prefix=f"constructing `{info.qualname}`: ")
    )
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
            function=info.qualname,
            hint=hint,
            note=_note(message, hint, statement),
            traceback=text,
        )
    ]


def check_attributes(
    module: Any,
    info: ClassInfo,
    path: str,
    variadic_rank: int,
    *,
    allow_effects: bool = False,
) -> list[Diagnostic]:
    """Construct the class once and compare `self.X` against its annotation."""
    _ensure_initialized()
    module_literals = _written_integers(module, max_value=None) if _written_integers else set()
    binder = DimBinder(variadic_rank=variadic_rank, literals=info.literal_dims | module_literals)
    attributes: list[Attribute] = []

    try:
        with active_guard(enabled=not allow_effects, phase="trace"):
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
            instance, binder = instantiate_retrying(info, cls, module, binder, chosen)
    except NotLive:
        return []
    except TraceFailed as exc:
        # A retry on divisible widths got past a guard to a real error.
        return _constructor_error(exc.error, path, info, exc.binder)
    except TraceSkipped as exc:
        pos = exc.position or info.position
        return [
            Diagnostic(
                path=path,
                line=pos.line,
                column=pos.column,
                end_line=pos.end_line,
                end_column=pos.end_column,
                rule=exc.rule,
                severity=_severity(exc.rule),
                message=exc.message,
                function=info.qualname,
                hint=exc.hint or None,
            )
        ]
    except Exception as exc:  # noqa: BLE001
        return _constructor_error(exc, path, info, binder)

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
    _ensure_initialized()
    hints = {
        name: result.binder.render_shape(shape) for name, shape in result.argument_shapes.items()
    }
    hints["return"] = describe(result.returned, result.binder)
    return hints


def _is_local(qualname: str) -> bool:
    return "<locals>" in qualname


def _check_one_file(path: str, job: dict[str, Any]) -> dict[str, Any]:
    variadic_rank = job.get("variadic_rank", 2)
    want_hover = job.get("hover", False)
    allow_effects = job.get("allow_effects", False)
    file_diags: list[dict[str, Any]] = []
    file_hovers: dict[str, dict[str, str]] = {}
    file_checked = 0
    file_skipped = 0
    saved_path = list(sys.path)
    buffer = job.get("sources", {}).get(path)
    if buffer is not None:
        _prime_linecache(path, buffer)
    # The project tree stays blocked even where it sits under a scratch
    # directory: without this, a project checked out under /tmp (pytest's
    # tmp_path included) could overwrite its own files through the temp-dir
    # allowance. Entered manually rather than with `with` so the existing
    # try/finally below can own the exit on every path out.
    _protection = active_guard(
        enabled=False, protect=job.get("protect_roots") or [str(Path(path).parent)]
    )
    _protection.__enter__()
    try:
        source = buffer if buffer is not None else Path(path).read_text(encoding="utf-8")
        scan = scan_source(source, path)
        if scan.syntax_error is not None:
            return {
                "diagnostics": [],
                "hovers": {},
                "checked_functions": 0,
                "skipped_functions": 0,
            }

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
        local_targets = [
            t for t in targets if t.owner is not None and t.owner.qualname in local_classes
        ]
        file_skipped += len(local_targets)
        targets = [t for t in targets if t.owner is None or t.owner.qualname not in local_classes]
        classes = [c for c in scan.classes if c.all_attributes or c.qualname in local_classes]
        if not targets and not classes:
            return {
                "diagnostics": [],
                "hovers": {},
                "checked_functions": 0,
                "skipped_functions": file_skipped,
            }

        # Threads started by the import are policed here, while the guard is
        # still installed: anything such a thread writes after the guard
        # uninstalls would otherwise escape coverage entirely. Only
        # constructor-spawned threads were checked before (in
        # tracing.instantiate); an import that leaves one running is reported
        # the same way, as an import error, instead of being left to write
        # into the gap between the import and trace phases.
        before_threads = set(threading.enumerate())
        try:
            with active_guard(enabled=not allow_effects, phase="import"):
                module = import_from_path(Path(path), buffer)
                if not allow_effects:
                    spawned = [
                        t for t in threading.enumerate() if t not in before_threads and t.is_alive()
                    ]
                    for thread in spawned:
                        attempt_cooperative_stop(thread)
                    surviving = [t for t in spawned if t.is_alive()]
                    if surviving:
                        names = ", ".join(f"`{t.name}`" for t in surviving)
                        file_diags.append(
                            Diagnostic(
                                path=path,
                                line=0,
                                column=0,
                                rule="import-error",
                                severity=Severity.ERROR,
                                message=f"module left background thread {names} running during import",
                                hint="shut down background threads at import time or start them lazily",
                            ).to_json()
                        )
                        file_skipped += len(targets)
                        return {
                            "diagnostics": file_diags,
                            "hovers": {},
                            "checked_functions": 0,
                            "skipped_functions": file_skipped,
                        }
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            diag = _blocked_diagnostic(
                exc, path, Position(0, 0, 0, 1), DimBinder(), rule="import-error"
            )
            if diag is None:
                position, text, _, statement = _anchor(exc, path, Position(0, 0, 0, 1), DimBinder())
                diag = Diagnostic(
                    path=path,
                    line=position.line,
                    column=position.column,
                    end_line=position.end_line,
                    end_column=position.end_column,
                    rule="import-error",
                    severity=Severity.ERROR,
                    message=_format_exc(exc, statement),
                    traceback=text,
                )
            file_diags.append(diag.to_json())
            file_skipped += len(targets)
            return {
                "diagnostics": file_diags,
                "hovers": {},
                "checked_functions": 0,
                "skipped_functions": file_skipped,
            }

        with active_guard(enabled=not allow_effects, phase="trace"):
            for info in classes:
                for diagnostic in check_attributes(
                    module, info, path, variadic_rank, allow_effects=allow_effects
                ):
                    file_diags.append(diagnostic.to_json())

            for target in targets:
                outcome = check_target(
                    module, target, path, variadic_rank, targets, allow_effects=allow_effects
                )
                found, result = outcome
                for diagnostic in found:
                    file_diags.append(diagnostic.to_json())
                if want_hover:
                    file_hovers[target.qualname] = (
                        shapes_for_hover(result) if result is not None else {}
                    )
                if getattr(outcome, "skipped", False):
                    file_skipped += 1
                elif getattr(outcome, "checked", False):
                    file_checked += 1
    finally:
        _clear_linecache([path])
        _restore_modules_and_path(saved_path)
        _protection.__exit__(None, None, None)

    return {
        "diagnostics": file_diags,
        "hovers": file_hovers,
        "checked_functions": file_checked,
        "skipped_functions": file_skipped,
    }


def _run_file_in_child(path: str, job: dict[str, Any]) -> dict[str, Any]:
    r_fd, w_fd = os.pipe()
    if sys.stdout is not None:
        sys.stdout.flush()
    if sys.stderr is not None:
        sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            res = _check_one_file(path, job)
            payload = json.dumps(res).encode("utf-8")
            total = 0
            while total < len(payload):
                written = os.write(w_fd, payload[total:])
                total += written
        except KeyboardInterrupt:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(130)
        except BaseException as exc:  # noqa: BLE001
            try:
                err_payload = json.dumps(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ).encode("utf-8")
                os.write(w_fd, err_payload)
            except Exception:  # noqa: BLE001, S110
                pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    # Parent process
    os.close(w_fd)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(r_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)

    _, status = os.waitpid(pid, 0)
    raw = b"".join(chunks)

    res: dict[str, Any] | None = None
    if raw:
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict) and "diagnostics" in parsed:
                res = parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            res = None

    if res is not None:
        return res

    detail = ""
    if raw:
        with contextlib.suppress(Exception):
            parsed_err = json.loads(raw.decode("utf-8"))
            if isinstance(parsed_err, dict) and "error" in parsed_err:
                detail = str(parsed_err["error"])

    if not detail:
        if os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            detail = f"signal {sig}"
        elif os.WIFEXITED(status):
            code = os.WEXITSTATUS(status)
            detail = f"exit code {code}"
        else:
            detail = "died unexpectedly"

    diag = Diagnostic(
        path=path,
        line=0,
        column=0,
        rule="worker-error",
        severity=_severity("worker-error"),
        message=f"worker child process failed ({detail})",
    )
    return {
        "diagnostics": [diag.to_json()],
        "hovers": {},
        "checked_functions": 0,
        "skipped_functions": 0,
    }


_TORCHTYC_DIR = Path(__file__).resolve().parent
_HEARTBEAT_SECONDS = 1.0
_EINOPS_RANK = re.compile(r"expected (\d+) dims\. Received (\d+)-dim tensor")


def _is_project_file(filename: str | None, root: Path) -> bool:
    if not filename or filename.startswith("<"):
        return False
    try:
        path = Path(filename).resolve()
    except (OSError, ValueError):
        return False
    if not path.is_relative_to(root) or path.is_relative_to(_TORCHTYC_DIR):
        return False
    return not is_system_path(path)


def _deepest_project_line(frame: Any, root: Path) -> tuple[str, int] | None:
    while frame is not None:
        if _is_project_file(frame.f_code.co_filename, root):
            return frame.f_code.co_filename, frame.f_lineno - 1
        frame = frame.f_back
    return None


def _run_as_main(path: Path) -> None:
    import runpy

    top, dotted = _module_name(path)
    if "." in dotted:
        if str(top) not in sys.path:
            sys.path.insert(0, str(top))
        runpy.run_module(dotted, run_name="__main__", alter_sys=True)
    else:
        sys.path.insert(0, str(path.parent))
        runpy.run_path(str(path), run_name="__main__")


def _names_on_line(statement: str | None) -> list[str]:
    if not statement:
        return []
    try:
        tree = ast.parse(statement.strip())
    except SyntaxError:
        return []
    seen: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in seen:
            seen.append(node.id)
    return seen


def _is_model_output(value: Any) -> bool:
    return any(cls.__name__ == "ModelOutput" for cls in type(value).__mro__)


def _run_got(frame: Any, statement: str | None, state: Any) -> str | None:
    parts: list[str] = []
    for name in _names_on_line(statement):
        value = frame.f_locals.get(name) if frame is not None else None
        if isinstance(value, torch.Tensor):
            parts.append(f"{name} {state.render_shape(tuple(value.shape))}")
        elif value is not None and _is_model_output(value):
            parts.append(f"{name} {type(value).__name__}")
    return "; ".join(parts) or None


def _run_hint(exc: BaseException, frame: Any, statement: str | None) -> str | None:
    for name in _names_on_line(statement):
        value = frame.f_locals.get(name) if frame is not None else None
        if value is not None and _is_model_output(value):
            hint = f"`{name}` is a {type(value).__name__}, not a tensor"
            if getattr(value, "logits", None) is not None:
                hint += f"; did you mean `{name}.logits`?"
            return hint
    match = _EINOPS_RANK.search(str(exc))
    if match:
        return f"the pattern names {match[1]} axes but the tensor has {match[2]}"
    return None


def _callee(frames: list[traceback.FrameSummary], index: int) -> str | None:
    """The function the user's line called, skipping torchtyc's own wrappers.

    The einops recorder sits between the user's `rearrange(...)` and einops,
    and its frame must not name the call.
    """
    for summary in frames[index + 1 :]:
        try:
            if Path(summary.filename).resolve().is_relative_to(_TORCHTYC_DIR):
                continue
        except (OSError, ValueError):
            pass
        if summary.name.startswith("_") or summary.name == "<module>":
            return None
        return summary.name
    return None


def _run_diagnostic(
    exc: BaseException, outcome: Any, entry_path: str, root: Path, state: Any
) -> Diagnostic:
    summaries = traceback.extract_tb(exc.__traceback__)
    frames = _live_frames(exc)
    mine = [i for i, s in enumerate(summaries) if _is_project_file(s.filename, root)]
    path, line, column, end_line, end_column = entry_path, 0, 0, 0, 1
    frame = statement = None
    if isinstance(exc, SyntaxError) and exc.filename and _is_project_file(exc.filename, root):
        path, line = exc.filename, max((exc.lineno or 1) - 1, 0)
        column, end_line, end_column = max((exc.offset or 1) - 1, 0), line, max(exc.offset or 1, 1)
    elif mine:
        chosen = summaries[mine[-1]]
        frame, statement = frames[mine[-1]], chosen.line
        path, line = chosen.filename, chosen.lineno - 1
        end_line = (chosen.end_lineno or chosen.lineno) - 1
        if chosen.colno is not None and chosen.end_colno is not None:
            column, end_column = chosen.colno, chosen.end_colno
        else:
            end_column = len(chosen.line or "") or 1
    via = " → ".join(f"{summaries[i].name} (line {summaries[i].lineno})" for i in mine) or None
    if outcome.rule == "run-error":
        last = (str(exc).strip().splitlines() or [""])[-1]
        callee = _callee(summaries, mine[-1]) if mine else None
        head = f"`{callee}` raised" if callee else "raised"
        message = f"{head} {type(exc).__name__}: {last}" if last else f"{head} {type(exc).__name__}"
    else:
        message = outcome.reason
    return Diagnostic(
        path=path,
        line=line,
        column=column,
        end_line=end_line,
        end_column=end_column,
        rule=outcome.rule,
        severity=_severity(outcome.rule),
        message=message,
        got=_run_got(frame, statement, state) if outcome.rule == "run-error" else None,
        hint=_run_hint(exc, frame, statement) if outcome.rule == "run-error" else None,
        via=via,
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def run_entry(entry: dict[str, Any], job: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Run one entry point on meta, emitting diagnostics as they are found."""
    _ensure_initialized()
    from . import metaruntime

    path = entry["path"]
    root = Path(job["root"]).resolve()
    status: dict[str, Any] = {
        "entry": entry["spec"],
        "state": "completed",
        "path": None,
        "line": None,
        "reason": None,
        "active": [],
        "inactive": [],
        "ignored": [],
    }
    saved_argv, saved_path = list(sys.argv), list(sys.path)
    main_ident = threading.get_ident()
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(_HEARTBEAT_SECONDS):
            where = _deepest_project_line(sys._current_frames().get(main_ident), root)
            emit(
                {
                    "event": "heartbeat",
                    "entry": entry["spec"],
                    "path": where[0] if where else None,
                    "line": where[1] if where else None,
                }
            )

    heartbeat = threading.Thread(target=beat, name="torchtyc-heartbeat", daemon=True)
    try:
        with (
            active_guard(enabled=False, protect=job.get("protect_roots") or [str(root)]),
            active_guard(enabled=not job.get("allow_effects", False), phase="run"),
            metaruntime.install(job.get("path_map", ())) as state,
        ):
            heartbeat.start()
            try:
                sys.argv = [path, *entry.get("args", [])]
                if entry.get("func"):
                    module = import_from_path(Path(path))
                    result = getattr(module, entry["func"])()
                    if inspect.iscoroutine(result):
                        asyncio.run(result)
                else:
                    _run_as_main(Path(path))
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001
                outcome = metaruntime.classify(exc)
                status["state"] = outcome.state
                if outcome.rule is not None:
                    diagnostic = _run_diagnostic(exc, outcome, path, root, state)
                    emit({"event": "diagnostic", "diagnostic": diagnostic.to_json()})
                    status.update(
                        path=diagnostic.path,
                        line=diagnostic.line,
                        reason=outcome.reason or None,
                    )
            status.update(
                active=list(state.active),
                inactive=list(state.inactive),
                ignored=sorted(state.ignored),
            )
    finally:
        stop.set()
        sys.argv = saved_argv
        sys.path[:] = saved_path
    return status


def _run_entry_in_child(entry: dict[str, Any], job: dict[str, Any], channel: Any) -> None:
    r_fd, w_fd = os.pipe()
    for stream in (sys.stdout, sys.stderr, channel):
        if stream is not None:
            stream.flush()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        lock = threading.Lock()

        def emit(message: dict[str, Any]) -> None:
            data = memoryview((json.dumps(message) + "\n").encode("utf-8"))
            with lock:
                while data:
                    data = data[os.write(w_fd, data) :]

        code = 0
        try:
            emit({"event": "entry_result", "status": run_entry(entry, job, emit)})
        except KeyboardInterrupt:
            code = 130
        except BaseException as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                emit(
                    {
                        "event": "entry_result",
                        "status": {
                            "entry": entry["spec"],
                            "state": "crashed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        },
                    }
                )
        finally:
            with contextlib.suppress(OSError):
                os.close(w_fd)
            os._exit(code)

    os.close(w_fd)
    finished = False
    with os.fdopen(r_fd, "r", encoding="utf-8") as stream:
        for line in stream:
            channel.write(line)
            channel.flush()
            finished = finished or '"entry_result"' in line
    _, wait_status = os.waitpid(pid, 0)
    if not finished:
        detail = (
            f"signal {os.WTERMSIG(wait_status)}"
            if os.WIFSIGNALED(wait_status)
            else f"exit code {os.WEXITSTATUS(wait_status)}"
        )
        diagnostic = Diagnostic(
            path=entry["path"],
            line=0,
            column=0,
            rule="worker-error",
            severity=_severity("worker-error"),
            message=f"the run's child process died ({detail})",
        )
        channel.write(
            json.dumps({"event": "diagnostic", "diagnostic": diagnostic.to_json()}) + "\n"
        )
        channel.write(
            json.dumps(
                {
                    "event": "entry_result",
                    "status": {"entry": entry["spec"], "state": "crashed", "reason": detail},
                }
            )
            + "\n"
        )
        channel.flush()


def _run_entries(job: dict[str, Any], channel: Any) -> dict[str, Any]:
    _ensure_initialized()
    channel = channel if channel is not None else sys.stderr
    lock = threading.Lock()

    def emit(message: dict[str, Any]) -> None:
        with lock:
            channel.write(json.dumps(message) + "\n")
            channel.flush()

    for entry in job["entries"]:
        emit({"event": "entry_start", "entry": entry["spec"]})
        if hasattr(os, "fork"):
            _run_entry_in_child(entry, job, channel)
        else:
            emit({"event": "entry_result", "status": run_entry(entry, job, emit)})
    emit({"event": "done"})
    return {"diagnostics": [], "hovers": {}, "checked_functions": 0, "skipped_functions": 0}


def run_job(job: dict[str, Any], channel: Any = None) -> dict[str, Any]:
    if job.get("mode") == "run":
        return _run_entries(job, channel)
    _ensure_initialized()
    diagnostics: list[dict[str, Any]] = []
    hovers: dict[str, dict[str, dict[str, str]]] = {}
    total_checked = 0
    total_skipped = 0
    can_fork = hasattr(os, "fork")

    for path in job["paths"]:
        if channel is not None:
            channel.write(json.dumps({"event": "file_start", "path": path}) + "\n")
            channel.flush()

        if can_fork:
            file_res = _run_file_in_child(path, job)
        else:
            file_res = _check_one_file(path, job)

        file_diags = file_res.get("diagnostics", [])
        file_hovers = file_res.get("hovers", {})
        file_checked = file_res.get("checked_functions", 0)
        file_skipped = file_res.get("skipped_functions", 0)

        total_checked += file_checked
        total_skipped += file_skipped
        diagnostics.extend(file_diags)
        if file_hovers:
            hovers[path] = file_hovers

        if channel is not None:
            channel.write(
                json.dumps(
                    {
                        "event": "file_result",
                        "path": path,
                        "diagnostics": file_diags,
                        "hovers": {path: file_hovers} if file_hovers else {},
                        "checked_functions": file_checked,
                        "skipped_functions": file_skipped,
                    }
                )
                + "\n"
            )
            channel.flush()

    if channel is not None:
        channel.write(json.dumps({"event": "done"}) + "\n")
        channel.flush()

    return {
        "diagnostics": diagnostics,
        "hovers": hovers,
        "checked_functions": total_checked,
        "skipped_functions": total_skipped,
    }


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

    streaming = job.get("stream", False)
    if streaming:
        channel.write(json.dumps({"event": "ready"}) + "\n")
        channel.flush()

    try:
        result = run_job(job, channel=channel if streaming else None)
    except Exception as exc:  # noqa: BLE001
        result = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "diagnostics": [],
            "hovers": {},
        }
        if streaming:
            channel.write(json.dumps(result) + "\n")
            channel.flush()
            return 1
        json.dump(result, channel)
        channel.flush()
        return 0

    if not streaming:
        json.dump(result, channel)
        channel.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
