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
import sys
import traceback
from pathlib import Path
from typing import Any

from .binding import DimBinder
from .diagnostics import RULES, Diagnostic, Severity
from .discovery import ClassInfo, Position, Target, scan_source
from .tracing import (
    NotLive,
    TraceFailed,
    TraceResult,
    TraceSkipped,
    check_return,
    describe,
    instantiate,
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


def _anchor(exc: BaseException, path: str, fallback: Position) -> tuple[Position, str]:
    """Point at the deepest frame inside the file being checked.

    A shape error usually surfaces several frames down, inside einsum or matmul.
    The line the user needs to see is the last one they wrote, not torch's.

    The span comes from the frame's column offsets rather than its source text,
    because `FrameSummary.line` is stripped of leading indentation: measuring it
    would start the underline inside the indent and stop short of the statement.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in frames if f.filename and Path(f.filename).resolve() == Path(path).resolve()]
    chosen = mine[-1] if mine else None
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if chosen is None:
        return fallback, text
    line = chosen.lineno - 1 if chosen.lineno else fallback.line
    end_line = chosen.end_lineno - 1 if chosen.end_lineno else line
    if chosen.colno is not None and chosen.end_colno is not None:
        return Position(line, chosen.colno, end_line, chosen.end_colno), text
    # No column information: fall back to underlining the whole line.
    return Position(line, 0, line, len(chosen.line or "") or 1), text


def _severity(rule: str) -> Severity:
    entry = RULES.get(rule)
    return entry.severity if entry else Severity.ERROR


def check_target(
    module: Any, target: Target, path: str, variadic_rank: int
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
        position, text = _anchor(exc.error, path, target.position)
        return [
            Diagnostic(
                path=path,
                line=position.line,
                column=position.column,
                end_line=position.end_line,
                end_column=position.end_column,
                rule="trace-error",
                severity=Severity.ERROR,
                message=exc.binder.rename_primes(f"{type(exc.error).__name__}: {exc.error}"),
                function=target.qualname,
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
    except Exception as exc:  # noqa: BLE001 - any user error is a finding
        position, text = _anchor(exc, path, target.position)
        return [
            Diagnostic(
                path=path,
                line=position.line,
                column=position.column,
                end_line=position.end_line,
                end_column=position.end_column,
                rule="trace-error",
                severity=Severity.ERROR,
                message=f"{type(exc).__name__}: {exc}",
                function=target.qualname,
                traceback=text,
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

    try:
        cls = resolve_qualname(
            module,
            info.qualname,
            conditional=info.conditional,
            span=(info.def_line, info.end_line),
        )
        instance = instantiate(info, cls, binder, info.dim_names)
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
        position, text = _anchor(exc, path, info.position)
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
    for attribute in info.attributes:
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
        # factory with several annotated methods does not repeat itself.
        local_classes = {c.qualname for c in scan.classes if _is_local(c.qualname)}
        targets = [t for t in targets if t.owner is None or t.owner.qualname not in local_classes]
        classes = [c for c in scan.classes if c.attributes or _is_local(c.qualname)]
        if not targets and not classes:
            continue

        try:
            module = import_from_path(Path(path), buffer)
        except Exception as exc:  # noqa: BLE001
            position, text = _anchor(exc, path, Position(0, 0, 0, 1))
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
            found, result = check_target(module, target, path, variadic_rank)
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
