"""Putting the passes together: lint in-process, trace in a subprocess.

Two passes with different costs and different failure modes:

  * the lint pass parses the file and reports what is visible in the source. It
    is fast, cannot fail, and works on a file that does not import.
  * the trace pass hands the file to a worker running the project interpreter.
    It is the one that knows real shapes, and the one that can time out.

A caller that only wants fast feedback can run the first and skip the second.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .annotations import ArraySpec, TupleSpec
from .config import Config
from .diagnostics import RULES, Diagnostic, Severity
from .discovery import FileScan, Suppression, Target, iter_arrays, scan_source
from .einops_rules import check_call

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# torchtyc is appended to sys.path rather than put on PYTHONPATH, because
# PYTHONPATH takes precedence over the interpreter's own site-packages. A
# torchtyc installed under one Python would otherwise shadow the project's
# torch and jaxtyping with its own. Appending means the project always wins,
# and one installation can check any number of virtualenvs.
_BOOTSTRAP = (
    "import sys; sys.path.append(sys.argv[1]); "
    "from torchtyc.worker import main; raise SystemExit(main())"
)


@dataclass
class Report:
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # path -> qualname -> traced shapes, as the worker returned them.
    hovers: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    checked_files: int = 0
    checked_functions: int = 0
    worker_error: str | None = None

    def shapes_in(self, path: str) -> dict[str, dict[str, str]]:
        """Every traced qualname in one file, under the path it was checked as."""
        return self.hovers.get(path, {})

    @property
    def errors(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity is Severity.ERROR)

    @property
    def ok(self) -> bool:
        return self.errors == 0 and self.worker_error is None


def lint_scan(scan: FileScan, config: Config) -> list[Diagnostic]:
    """Everything decidable from the source text alone."""
    out: list[Diagnostic] = []
    path = scan.path

    if scan.syntax_error is not None:
        message, position = scan.syntax_error
        return [
            Diagnostic(
                path=path,
                line=position.line,
                column=position.column,
                end_line=position.end_line,
                end_column=position.end_column,
                rule="trace-error",
                severity=Severity.ERROR,
                message=f"SyntaxError: {message}",
            )
        ]

    file_uses_jaxtyping = any(t.has_array_annotation for t in scan.targets)

    # A dimension name is shared by everything in its class: the constructor
    # parameters, the annotated attributes, and every method. Counting one
    # signature at a time would call `d_in` unused in a forward that gets it
    # from `self.W`.
    scopes = _name_counts(scan)

    for target in scan.targets:
        scope = scopes[_scope_key(target)]
        out.extend(_lint_target(target, path, file_uses_jaxtyping, scope))
        if config.einops:
            for call in target.einops_calls:
                out.extend(check_call(call, target, path))

    return out


def _scope_key(target: Target) -> str:
    """A method shares its class's dimension names; a plain function shares nobody's.

    A method legitimately gets a dimension from `self.W` or from an `__init__`
    parameter, so the whole class is one scope. Two unrelated module-level
    functions share nothing, so each gets its own scope and one function reusing
    a name cannot mask another's unconstrained dimension.
    """
    return target.owner.qualname if target.owner else f"\0{target.qualname}"


def _may_supply_dim(param) -> bool:
    """Whether the tracer would bind this parameter to a dimension of that name."""
    return (
        bool(param.name)
        and not isinstance(param.spec, (ArraySpec, TupleSpec))
        and param.plain_type in (None, "int")
    )


def _name_counts(scan: FileScan) -> dict[str, dict[str, int]]:
    """How many times each dimension name is written, per class and per free function."""
    scopes: dict[str, dict[str, int]] = {}

    def add(scope: dict[str, int], spec) -> None:
        for array in iter_arrays(spec):
            for dim in array.dims:
                if dim.kind in ("named", "variadic") and dim.name:
                    scope[dim.name] = scope.get(dim.name, 0) + 1

    for info in scan.classes:
        scope = scopes.setdefault(info.qualname, {})
        for attribute in info.all_attributes:
            add(scope, attribute.spec)
        for param in info.all_init_params:
            if param.name:
                # A constructor parameter that names a dimension counts as a use,
                # because that is where the dimension gets its size.
                scope[param.name] = scope.get(param.name, 0) + 1

    for target in scan.targets:
        scope = scopes.setdefault(_scope_key(target), {})
        for param in target.params:
            add(scope, param.spec)
            if _may_supply_dim(param):
                # An integer parameter that names a dimension is where that
                # dimension gets its size, exactly as in a constructor.
                scope[param.name] = scope.get(param.name, 0) + 1
        add(scope, target.returns)

    return scopes


def _lint_target(
    target: Target, path: str, file_uses_jaxtyping: bool, scope: dict[str, int]
) -> list[Diagnostic]:
    out: list[Diagnostic] = []

    def at(position, rule: str, message: str, **extra) -> Diagnostic:
        return Diagnostic(
            path=path,
            line=position.line,
            column=position.column,
            end_line=position.end_line,
            end_column=position.end_column,
            rule=rule,
            severity=RULES[rule].severity,
            message=message,
            function=target.qualname,
            **extra,
        )

    if target.annotation_error:
        out.append(
            at(
                target.returns_position or target.position,
                "unsupported-annotation",
                f"return annotation: {target.annotation_error}",
            )
        )
    for param in target.params:
        if param.annotation_error:
            out.append(
                at(
                    param.position,
                    "unsupported-annotation",
                    f"`{param.name}`: {param.annotation_error}",
                )
            )

    if not target.has_array_annotation:
        if file_uses_jaxtyping and target.name == "forward":
            out.append(
                at(
                    target.position,
                    "missing-annotation",
                    f"`{target.qualname}` has no jaxtyping annotation, so it is not shape checked",
                    hint="annotate its tensor arguments and return to bring it under the checker",
                )
            )
        return out

    if target.returns is None:
        out.append(
            at(
                target.position,
                "anonymous-return",
                f"`{target.qualname}` annotates its arguments but not its return",
                hint="the return is where most shape bugs show up",
            )
        )

    local: set[str] = set()
    specs = [p.spec for p in target.params] + [target.returns]
    for spec in specs:
        for array in iter_arrays(spec):
            for dim in array.dims:
                if dim.kind in ("named", "variadic") and dim.name:
                    local.add(dim.name)

    for name in sorted(local):
        if scope.get(name, 0) == 1:
            out.append(
                at(
                    target.position,
                    "unused-dim",
                    f"dimension `{name}` appears once, so it constrains nothing",
                    hint="use `_` if it is genuinely free, or reuse the name where it must match",
                )
            )

    return out


class WorkerCancelled(Exception):
    """The worker was stopped because nobody wants its result anymore.

    The language server raises this through run_worker when a newer trace
    supersedes the run, or the file closes underneath it. It says nothing
    about the checked code, so it never becomes a diagnostic. The stale run
    just ends and publishes nothing.
    """


# How often a running worker checks whether it is still wanted. Short enough
# that a superseded trace dies before the user notices, long enough that the
# wait costs nothing next to a torch import.
_WORKER_POLL_SECONDS = 0.05
# How long a terminate request gets before it becomes a kill. A worker in the
# middle of importing torch can take a moment to unwind.
_WORKER_TERMINATE_GRACE_SECONDS = 2.0


def terminate_worker(proc: subprocess.Popen[str]) -> None:
    """Send SIGTERM to the worker's process group and return at once.

    This is the event-loop side of stopping a worker: it never blocks and
    never raises, so a keystroke is never held up by a dying child. The
    worker thread follows up with a grace period, a kill, and the wait that
    reaps the child (see _finish). A missing process just means the child
    already exited.
    """
    try:
        if os.name == "posix" and proc.pid is not None:
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except OSError:
        try:
            proc.terminate()
        except OSError:
            pass


def _kill_worker(proc: subprocess.Popen[str]) -> None:
    """Send SIGKILL to the worker's process group and return at once."""
    try:
        if os.name == "posix" and proc.pid is not None:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _finish(proc: subprocess.Popen[str]) -> None:
    """Stop a worker the caller no longer wants, leaving no zombie behind.

    Runs on the worker thread, which may block: terminate first so a
    cooperative worker exits cleanly, kill after a short grace so a stuck one
    still dies, then wait so the child is reaped either way.
    """
    try:
        terminate_worker(proc)
        try:
            proc.wait(timeout=_WORKER_TERMINATE_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        except OSError:
            return
        _kill_worker(proc)
        try:
            proc.wait(timeout=_WORKER_TERMINATE_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            pass
    finally:
        for name in ("stdin", "stdout", "stderr"):
            pipe = getattr(proc, name, None)
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass


def _communicate(
    proc: subprocess.Popen[str],
    payload: str,
    timeout: float | None,
    cancel: threading.Event | None,
) -> tuple[str | None, str | None]:
    """Collect the worker's output, stopping early when asked.

    Returns the two pipes, or (None, None) when the timeout runs out first.
    Raises WorkerCancelled when `cancel` is set, discarding whatever the
    child wrote: a kill mid-write leaves half a JSON document on the pipe,
    and parsing that would report a result that never happened.

    The payload goes in on the first wait only. Retrying the wait after a
    timeout keeps what the child wrote so far, so later waits pass no input.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    pending: str | None = payload
    while True:
        if cancel is not None and cancel.is_set():
            _finish(proc)
            raise WorkerCancelled()
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            _finish(proc)
            return None, None
        quantum = (
            _WORKER_POLL_SECONDS if remaining is None else min(_WORKER_POLL_SECONDS, remaining)
        )
        try:
            if pending is not None:
                stdout, stderr = proc.communicate(input=pending, timeout=quantum)
            else:
                stdout, stderr = proc.communicate(timeout=quantum)
        except subprocess.TimeoutExpired:
            pending = None
            continue
        if cancel is not None and cancel.is_set():
            # Stopped from another thread while the wait returned. The output
            # ends wherever the kill landed, so it is not a result either.
            _finish(proc)
            raise WorkerCancelled()
        return stdout, stderr


def run_worker(
    paths: list[str],
    config: Config,
    sources: dict[str, str] | None = None,
    hover: bool = False,
    *,
    cancel: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen[str]], None] | None = None,
) -> tuple[list[Diagnostic], dict[str, dict[str, dict[str, str]]], str | None]:
    """Trace the files in a subprocess and bring back what it found.

    The worker runs with the project root as its working directory, which is
    not the caller's, so a relative path would resolve against the wrong
    directory there. Paths go over as absolute and come back mapped to the
    strings the caller passed, since those are what the rest of the report is
    keyed by.

    `cancel` lets the language server stop a worker a newer trace no longer
    needs. The plain check never passes it, so it runs exactly as before,
    including the timeout. `on_proc` receives the child right after it
    spawns, so the server can also stop it directly while the worker thread
    is still starting up.
    """
    absolute = {str(Path(p).resolve()): p for p in paths}
    job = {
        "paths": list(absolute),
        "variadic_rank": config.variadic_rank,
        "sources": {str(Path(p).resolve()): text for p, text in (sources or {}).items()},
        "hover": hover,
    }

    env = dict(os.environ)
    if config.extra_paths:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([p for p in [*config.extra_paths, existing] if p])
    # Importing user code must not leave .pyc files in the project tree.
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        proc = subprocess.Popen(
            [config.interpreter, "-c", _BOOTSTRAP, str(_PACKAGE_ROOT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(config.root),
            env=env,
            # The worker imports project code, which can spawn children of its
            # own. Its own group means stopping it stops those too.
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        return [], {}, f"could not start {config.interpreter}: {exc}"
    if on_proc is not None:
        on_proc(proc)

    stdout, stderr = _communicate(proc, json.dumps(job), config.timeout, cancel)
    if stdout is None:
        return [], {}, f"the trace timed out after {config.timeout:g}s"

    if not stdout.strip():
        detail = (stderr or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {proc.returncode}"
        return [], {}, f"the worker produced no output ({tail})"

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return [], {}, "the worker produced output that was not JSON"

    diagnostics = []
    for entry in result.get("diagnostics", []):
        diagnostic = Diagnostic.from_json(entry)
        diagnostic.path = absolute.get(diagnostic.path, diagnostic.path)
        diagnostics.append(diagnostic)
    hovers = {absolute.get(p, p): shapes for p, shapes in result.get("hovers", {}).items()}
    return diagnostics, hovers, result.get("error")


def apply_suppressions(
    diagnostics: list[Diagnostic], suppressions: list[Suppression]
) -> list[Diagnostic]:
    by_line: dict[int, list[Suppression]] = {}
    for suppression in suppressions:
        by_line.setdefault(suppression.line, []).append(suppression)

    kept: list[Diagnostic] = []
    for diagnostic in diagnostics:
        matches = [
            s
            for line in range(diagnostic.line, (diagnostic.end_line or diagnostic.line) + 1)
            for s in by_line.get(line, [])
            if s.rules is None or diagnostic.rule in s.rules
        ]
        if matches:
            for match in matches:
                match.used = True
            continue
        kept.append(diagnostic)
    return kept


def check_paths(
    paths: list[str],
    config: Config,
    sources: dict[str, str] | None = None,
    hover: bool = False,
    *,
    cancel: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen[str]], None] | None = None,
) -> Report:
    report = Report()
    sources = sources or {}
    scans: dict[str, FileScan] = {}

    for path in paths:
        try:
            # An empty buffer is still the buffer: `or` would fall back to the
            # file on disk and report code the user has just deleted.
            text = sources[path] if path in sources else Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            report.diagnostics.append(
                Diagnostic(
                    path=path,
                    line=0,
                    column=0,
                    rule="import-error",
                    severity=Severity.ERROR,
                    message=str(exc),
                )
            )
            continue
        scans[path] = scan_source(text, path)

    traceable: list[str] = []
    for path, scan in scans.items():
        report.checked_files += 1
        report.checked_functions += sum(1 for t in scan.targets if t.has_array_annotation)
        report.diagnostics.extend(lint_scan(scan, config))
        needs_trace = any(t.has_array_annotation for t in scan.targets) or any(
            c.all_attributes for c in scan.classes
        )
        if scan.syntax_error is None and needs_trace:
            traceable.append(path)

    if traceable:
        traced, hovers, error = run_worker(
            traceable, config, sources, hover, cancel=cancel, on_proc=on_proc
        )
        report.diagnostics.extend(traced)
        report.hovers = hovers
        report.worker_error = error

    per_file: dict[str, list[Diagnostic]] = {}
    for diagnostic in report.diagnostics:
        per_file.setdefault(diagnostic.path, []).append(diagnostic)

    final: list[Diagnostic] = []
    for path in {*per_file, *scans}:
        items = per_file.get(path, [])
        scan = scans.get(path)
        if scan is None:
            final.extend(items)
            continue
        final.extend(apply_suppressions(items, scan.suppressions))
        for suppression in scan.suppressions:
            if not suppression.used and suppression.rules is not None:
                final.append(
                    Diagnostic(
                        path=path,
                        line=suppression.line,
                        column=0,
                        rule="suppression-unused",
                        severity=Severity.INFO,
                        message="this torchtyc: ignore matched no diagnostic",
                    )
                )

    final = [d for d in final if d.rule not in config.ignore and d.severity <= config.severity]
    final.sort(key=lambda d: (d.path, d.line, d.column, d.rule))
    report.diagnostics = final
    return report


def collect_files(targets: list[str], config: Config) -> list[str]:
    """Expand paths and directories into the python files worth checking."""
    found: list[str] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                # Only what is below the directory the user asked for: a
                # checkout that happens to live under `build` is not excluded.
                if any(part in config.exclude for part in child.relative_to(path).parts):
                    continue
                found.append(str(child))
        elif path.suffix == ".py":
            found.append(str(path))
    return found
