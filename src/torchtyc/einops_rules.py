"""Static checks on einops pattern strings.

These need no torch and no import, so they run in the editor's process and
survive a file that does not even import.

The error-level rules are deliberately narrow: each one flags something einops
itself rejects at runtime, so an error here is never a matter of taste. The
near-miss rule is the exception and is warning-level for that reason: einops
accepts an axis name one edit away from an annotated dimension quite happily,
and torchtyc only points out that the two probably meant to be the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .diagnostics import Diagnostic, Severity
from .discovery import EinopsCall, Target

_AXIS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# reduce takes the reduction as a positional argument, so its pattern is not
# the only string in the call.
_ONE_SIDED = {"pack", "unpack"}


@dataclass
class Pattern:
    left: list[list[str]]
    right: list[str]


class PatternError(ValueError):
    pass


def parse_pattern(text: str) -> Pattern:
    if "->" not in text:
        raise PatternError("the pattern has no `->`")
    if text.count("->") > 1:
        raise PatternError("the pattern has more than one `->`")

    left_text, right_text = text.split("->")
    left = [_axes(group) for group in left_text.split(",")]
    right = _axes(right_text)
    return Pattern(left=left, right=right)


def _axes(text: str) -> list[str]:
    if text.count("(") != text.count(")"):
        raise PatternError("unbalanced parentheses")
    return _AXIS.findall(text)


def check_call(call: EinopsCall, target: Target, path: str) -> list[Diagnostic]:
    if call.pattern is None:
        return []

    position = call.position
    base = {
        "path": path,
        "line": position.line,
        "column": position.column,
        "end_line": position.end_line,
        "end_column": position.end_column,
        "function": target.qualname,
    }

    if call.func in _ONE_SIDED:
        return []

    try:
        pattern = parse_pattern(call.pattern)
    except PatternError as exc:
        return [
            Diagnostic(
                **base,
                rule="einops-pattern",
                severity=Severity.ERROR,
                message=f"einops.{call.func}: {exc}",
                got=call.pattern,
            )
        ]

    out: list[Diagnostic] = []

    if call.func == "einsum" and not call.starred_args and len(pattern.left) != call.tensor_args:
        out.append(
            Diagnostic(
                **base,
                rule="einops-pattern",
                severity=Severity.ERROR,
                message=(
                    f"einops.einsum: the pattern describes {len(pattern.left)} operands "
                    f"but {call.tensor_args} tensors were passed"
                ),
                expected=f"{call.tensor_args} comma-separated groups",
                got=call.pattern,
            )
        )

    if call.func != "einsum" and len(pattern.left) > 1:
        out.append(
            Diagnostic(
                **base,
                rule="einops-pattern",
                severity=Severity.ERROR,
                message=f"einops.{call.func}: only einsum takes comma-separated operands",
                got=call.pattern,
            )
        )

    known = {axis for group in pattern.left for axis in group} | set(call.keywords)
    for axis in pattern.right:
        if axis not in known:
            out.append(
                Diagnostic(
                    **base,
                    rule="einops-unknown-axis",
                    severity=Severity.WARNING,
                    message=(
                        f"einops.{call.func}: `{axis}` appears on the right of `->` "
                        "but is neither an input axis nor a keyword argument"
                    ),
                    hint=f"pass its length, as in {call.func}(..., {axis}=<size>)",
                    got=call.pattern,
                )
            )

    # An axis name shared with the annotations is the common case, and a name
    # that is one character away from an annotated one is usually a typo.
    dim_names = target.dim_names
    if dim_names:
        for axis in sorted(known - set(call.keywords)):
            if axis in dim_names:
                continue
            close = sorted(name for name in dim_names if _one_edit_apart(axis, name))
            if close:
                out.append(
                    Diagnostic(
                        **base,
                        rule="einops-unknown-axis",
                        severity=Severity.WARNING,
                        message=(
                            f"einops.{call.func}: axis `{axis}` is one character from "
                            f"annotated dimension `{close[0]}`"
                        ),
                        hint="einops axis names are local, so this is only a warning",
                    )
                )

    return out


def _one_edit_apart(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 3:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for index in range(len(long)):
        if long[:index] + long[index + 1 :] == short:
            return True
    return False
