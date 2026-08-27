"""Rendering a report for a terminal, a machine, or a CI annotation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .diagnostics import Diagnostic, Severity
from .engine import Report

_COLORS = {
    Severity.ERROR: "\033[31m",
    Severity.WARNING: "\033[33m",
    Severity.INFO: "\033[36m",
}
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def use_color(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _relative(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return path


def render(report: Report, style: str, root: Path, color: bool | None = None) -> str:
    if color is None:
        color = use_color()
    if style == "json":
        return _json(report)
    if style == "github":
        return _github(report, root)
    if style == "concise":
        return _concise(report, root, color)
    return _full(report, root, color)


def _paint(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


def _concise(report: Report, root: Path, color: bool) -> str:
    lines = [
        f"{_relative(d.path, root)}:{d.line + 1}:{d.column + 1}: "
        f"{_paint(d.severity.label, _COLORS[d.severity], color)}[{d.rule}] {d.message}"
        for d in report.diagnostics
    ]
    lines.append(_summary(report, color))
    return "\n".join(lines)


def _full(report: Report, root: Path, color: bool) -> str:
    blocks: list[str] = []
    for d in report.diagnostics:
        head = (
            f"{_paint(_relative(d.path, root), _BOLD, color)}:{d.line + 1}:{d.column + 1}: "
            f"{_paint(f'{d.severity.label}[{d.rule}]', _COLORS[d.severity], color)}"
        )
        body = [head, f"  {d.message}"]
        if d.expected is not None or d.got is not None:
            width = max(len(str(d.expected or "")), len(str(d.got or "")))
            if d.expected is not None:
                body.append(f"    Expected: {d.expected:<{width}}")
            if d.got is not None:
                body.append(f"    Got:      {d.got:<{width}}")
        source = _source_line(d)
        if source:
            body.append(_paint(f"    {d.line + 1} | {source}", _DIM, color))
        if d.hint:
            body.append(_paint(f"  hint: {d.hint}", _DIM, color))
        blocks.append("\n".join(body))

    blocks.append(_summary(report, color))
    return "\n\n".join(blocks)


def _source_line(d: Diagnostic) -> str | None:
    try:
        lines = Path(d.path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if 0 <= d.line < len(lines):
        return lines[d.line].strip()
    return None


def _summary(report: Report, color: bool) -> str:
    if report.worker_error:
        return _paint(f"worker failed: {report.worker_error}", _COLORS[Severity.ERROR], color)

    counts = {level: 0 for level in Severity}
    for d in report.diagnostics:
        counts[d.severity] += 1

    if not report.diagnostics:
        return _paint(
            f"No problems in {report.checked_functions} function(s) "
            f"across {report.checked_files} file(s)",
            _COLORS[Severity.INFO],
            color,
        )

    parts = [f"{counts[level]} {level.label}(s)" for level in Severity if counts[level]]
    return (
        f"Found {', '.join(parts)} in {report.checked_functions} function(s) "
        f"across {report.checked_files} file(s)"
    )


def _json(report: Report) -> str:
    return json.dumps(
        {
            "diagnostics": [d.to_json() for d in report.diagnostics],
            "hovers": report.hovers,
            "checked_files": report.checked_files,
            "checked_functions": report.checked_functions,
            "worker_error": report.worker_error,
            "ok": report.ok,
        },
        indent=2,
    )


def _github(report: Report, root: Path) -> str:
    """GitHub Actions workflow commands, which become inline PR annotations."""
    levels = {Severity.ERROR: "error", Severity.WARNING: "warning", Severity.INFO: "notice"}
    lines: list[str] = []
    for d in report.diagnostics:
        message = d.message.replace("\n", "%0A")
        if d.hint:
            message += f"%0Ahint: {d.hint}"
        lines.append(
            f"::{levels[d.severity]} file={_relative(d.path, root)},"
            f"line={d.line + 1},col={d.column + 1},title=torchtyc[{d.rule}]::{message}"
        )
    if report.worker_error:
        lines.append(f"::error title=torchtyc::{report.worker_error}")
    return "\n".join(lines)
