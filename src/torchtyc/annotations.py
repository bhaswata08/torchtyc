"""Parsing of jaxtyping annotations out of source text.

Everything here works on `ast` nodes and strings only. It never imports torch or
evaluates the annotation, so the discovery pass can run in the editor's process
on a file that does not even import cleanly.

The grammar follows jaxtyping's dim strings:

    "batch seq d_model"   named dimensions
    "3 seq"               a fixed size
    "..."                 an anonymous run of dimensions
    "*batch"              a named run of dimensions
    "#channels"           a broadcastable dimension
    "_"                   one anonymous dimension
    "d_in+d_out"          a symbolic expression over already-bound names
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# jaxtyping's dtype classes. The values are resolved to concrete torch dtypes in
# the worker; here they are only names, so this table stays importable anywhere.
DTYPE_NAMES = frozenset(
    {
        "Shaped",
        "Num",
        "Inexact",
        "Real",
        "Float",
        "Complex",
        "Integer",
        "Int",
        "UInt",
        "Bool",
        "Key",
        "Float16",
        "Float32",
        "Float64",
        "BFloat16",
        "Float8e4m3fn",
        "Float8e5m2",
        "Complex64",
        "Complex128",
        "Int4",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt4",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    }
)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SYMBOLIC = re.compile(r"[+\-*/()]")


@dataclass(frozen=True)
class Dim:
    """One entry in a dim string."""

    kind: str  # "named" | "fixed" | "anonymous" | "variadic" | "symbolic"
    name: str | None = None
    size: int | None = None
    expr: str | None = None
    broadcastable: bool = False

    def __str__(self) -> str:
        prefix = "#" if self.broadcastable else ""
        if self.kind == "fixed":
            return str(self.size)
        if self.kind == "anonymous":
            return "_"
        if self.kind == "variadic":
            return "..." if self.name is None else f"*{self.name}"
        if self.kind == "symbolic":
            return f"{prefix}{self.expr}"
        return f"{prefix}{self.name}"


@dataclass(frozen=True)
class ArraySpec:
    """A single `Dtype[ArrayType, "dims"]` annotation."""

    dtype: str
    array_type: str
    dims: tuple[Dim, ...]
    raw: str
    # The dim string exactly as written, whitespace included. jaxtyping strips
    # it, so it changes no meaning, but `Float[Tensor, " d_model"]` is how a
    # single-axis annotation is written to keep ruff from reading it as a
    # forward reference (UP037). A suggestion has to write it back the same way.
    dim_text: str = ""

    @property
    def named_dims(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dims if d.name is not None)

    def __str__(self) -> str:
        return f'{self.dtype}[{self.array_type}, "{" ".join(str(d) for d in self.dims)}"]'

    def shape_str(self) -> str:
        return "(" + ", ".join(str(d) for d in self.dims) + ")"


@dataclass(frozen=True)
class TupleSpec:
    """A `tuple[...]` of array specs in a return position."""

    items: tuple[Spec, ...]

    def __str__(self) -> str:
        return "tuple[" + ", ".join(str(i) for i in self.items) + "]"


@dataclass(frozen=True)
class OpaqueSpec:
    """An annotation torchtyc does not model, kept so callers can report it."""

    raw: str

    def __str__(self) -> str:
        return self.raw


Spec = ArraySpec | TupleSpec | OpaqueSpec


class AnnotationError(ValueError):
    """The annotation looked like jaxtyping but could not be parsed."""


def parse_dim_string(text: str) -> tuple[Dim, ...]:
    """Split a jaxtyping dim string into dims, left to right."""
    dims: list[Dim] = []
    for token in text.split():
        dims.append(_parse_dim_token(token))
    return tuple(dims)


def _parse_dim_token(token: str) -> Dim:
    broadcastable = token.startswith("#")
    if broadcastable:
        token = token[1:]
    if token == "...":
        return Dim("variadic")
    if token.startswith("*"):
        rest = token[1:]
        if rest and not _IDENT.match(rest):
            raise AnnotationError(f"bad variadic dimension {token!r}")
        return Dim("variadic", name=rest or None, broadcastable=broadcastable)
    if token == "_":
        return Dim("anonymous", broadcastable=broadcastable)
    if token.isdigit():
        return Dim("fixed", size=int(token), broadcastable=broadcastable)
    if _SYMBOLIC.search(token):
        return Dim("symbolic", expr=token, broadcastable=broadcastable)
    if _IDENT.match(token):
        return Dim("named", name=token, broadcastable=broadcastable)
    raise AnnotationError(f"bad dimension {token!r}")


def _dotted_name(node: ast.expr) -> str | None:
    """Render `Tensor` or `torch.Tensor` or `nn.Parameter` back to a string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def parse_annotation(node: ast.expr | None) -> Spec | None:
    """Turn an annotation node into a spec, or None if there is no annotation.

    Anything that is not recognisably jaxtyping comes back as an OpaqueSpec so
    the caller can decide whether to warn or stay quiet.
    """
    if node is None:
        return None

    if isinstance(node, ast.Subscript):
        head = _dotted_name(node.value)
        tail = head.rsplit(".", 1)[-1] if head else None

        if tail in DTYPE_NAMES:
            return _parse_array(node, tail)

        if tail in ("tuple", "Tuple"):
            items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            parsed = tuple(parse_annotation(item) or OpaqueSpec("?") for item in items)
            if any(isinstance(p, ArraySpec) for p in parsed):
                return TupleSpec(parsed)

    return OpaqueSpec(ast.unparse(node))


def _parse_array(node: ast.Subscript, dtype: str) -> Spec:
    slice_node = node.slice
    if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
        raise AnnotationError(f'expected {dtype}[ArrayType, "dims"]')

    array_node, dims_node = slice_node.elts
    array_type = _dotted_name(array_node) or ast.unparse(array_node)

    if not isinstance(dims_node, ast.Constant) or not isinstance(dims_node.value, str):
        raise AnnotationError("the second argument must be a literal dim string")

    return ArraySpec(
        dtype=dtype,
        array_type=array_type,
        dims=parse_dim_string(dims_node.value),
        raw=ast.unparse(node),
        dim_text=dims_node.value,
    )
