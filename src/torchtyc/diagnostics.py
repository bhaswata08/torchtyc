"""Diagnostic records and the rule table.

A rule name is part of the tool's public surface: it appears in CLI output, in
`# torchtyc: ignore[rule]` comments, and in editor diagnostics. Renaming one is
a breaking change, so the table below is the single place they are defined.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    ERROR = 0
    WARNING = 1
    INFO = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, text: str) -> Severity:
        try:
            return cls[text.strip().upper()]
        except KeyError:
            raise ValueError(f"unknown severity {text!r}") from None


@dataclass(frozen=True)
class Rule:
    name: str
    severity: Severity
    summary: str


def _rule(name: str, severity: Severity, summary: str) -> Rule:
    rule = Rule(name, severity, summary)
    RULES[name] = rule
    return rule


RULES: dict[str, Rule] = {}

# Shape and dtype conclusions drawn from a successful trace.
SHAPE_MISMATCH = _rule(
    "shape-mismatch", Severity.ERROR, "a traced shape disagrees with its annotation"
)
RANK_MISMATCH = _rule(
    "rank-mismatch", Severity.ERROR, "a traced value has a different number of dimensions"
)
DTYPE_MISMATCH = _rule(
    "dtype-mismatch", Severity.ERROR, "a traced dtype is outside the annotated dtype set"
)
DIM_INCONSISTENT = _rule(
    "dim-inconsistent", Severity.ERROR, "one dimension name is bound to two different sizes"
)
NOT_A_TENSOR = _rule(
    "not-a-tensor", Severity.ERROR, "an annotated tensor position received a non-tensor"
)
TUPLE_ARITY = _rule(
    "tuple-arity", Severity.ERROR, "a tuple return has a different length than annotated"
)
ATTRIBUTE_MISMATCH = _rule(
    "attribute-mismatch",
    Severity.ERROR,
    "an annotated attribute on self holds a different shape after __init__",
)
DEVICE_MISMATCH = _rule("device-mismatch", Severity.WARNING, "a traced value left the meta device")

# Problems reaching the point where a shape could be compared.
TRACE_ERROR = _rule("trace-error", Severity.ERROR, "the function raised while being traced")
IMPORT_ERROR = _rule("import-error", Severity.ERROR, "the module could not be imported")
UNINSTANTIABLE = _rule(
    "uninstantiable", Severity.WARNING, "a module's __init__ could not be called automatically"
)
UNRESOLVED_ARG = _rule(
    "unresolved-arg", Severity.WARNING, "a parameter has no annotation and no default"
)
UNSUPPORTED_ANNOTATION = _rule(
    "unsupported-annotation", Severity.WARNING, "an annotation could not be parsed"
)

# Hygiene rules, off the critical path but cheap once a trace exists.
MISSING_ANNOTATION = _rule(
    "missing-annotation", Severity.INFO, "a public function has no jaxtyping annotation"
)
UNUSED_DIM = _rule(
    "unused-dim", Severity.INFO, "a dimension name is used exactly once, so it constrains nothing"
)
ANONYMOUS_RETURN = _rule(
    "anonymous-return", Severity.INFO, "a return annotation is missing while arguments have one"
)

# einops integration.
EINOPS_PATTERN = _rule(
    "einops-pattern", Severity.ERROR, "an einops pattern disagrees with the tensor it is given"
)
EINOPS_UNKNOWN_AXIS = _rule(
    "einops-unknown-axis", Severity.WARNING, "an einops axis name matches no annotated dimension"
)

SUPPRESSION_UNUSED = _rule(
    "suppression-unused", Severity.INFO, "a torchtyc: ignore comment matched no diagnostic"
)


@dataclass
class Diagnostic:
    """One finding, anchored to a file position.

    `line` and `column` are 0-based, which is what LSP expects. The CLI adds one
    when it prints them.
    """

    path: str
    line: int
    column: int
    rule: str
    message: str
    severity: Severity = Severity.ERROR
    end_line: int | None = None
    end_column: int | None = None
    function: str | None = None
    expected: str | None = None
    got: str | None = None
    hint: str | None = None
    traceback: str | None = None
    related: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.label
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Diagnostic:
        data = dict(data)
        data["severity"] = Severity.parse(data["severity"])
        return cls(**data)
