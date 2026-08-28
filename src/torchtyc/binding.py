"""Binding dimension names to sizes, and matching shapes against specs.

Every distinct dimension name gets its own prime number, starting at 101. Two
properties make that worth the trouble:

  * Distinct names always produce distinct sizes, so a bug that swaps `d_in`
    for `d_out` cannot hide behind two dimensions that happened to be equal.
  * Products stay unambiguous. If a traced dimension comes out as 10403 and the
    annotation says one dimension, factoring 101 * 103 tells you the function
    flattened two axes together, and the message can say so.

Sizes above a few hundred cost nothing here because no memory is ever allocated:
the tracer runs on meta tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .annotations import ArraySpec, Dim

FIRST_PRIME = 101
# How many concrete dimensions an unnamed `...` run stands for while tracing.
# Two is enough to catch code that assumes a single batch dimension, and small
# enough that a rank error stays readable.
DEFAULT_VARIADIC_RANK = 2


def _primes_from(start: int, count: int) -> list[int]:
    found: list[int] = []
    candidate = start
    while len(found) < count:
        if all(candidate % p for p in range(2, int(candidate**0.5) + 1)):
            found.append(candidate)
        candidate += 1
    return found


_PRIME_POOL = _primes_from(FIRST_PRIME, 512)


class BindingError(ValueError):
    """A shape could not be matched against a spec."""

    def __init__(
        self,
        message: str,
        *,
        rule: str = "shape-mismatch",
        expected: str = "",
        got: str = "",
        hint: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.rule = rule
        self.expected = expected
        self.got = got
        self.hint = hint


@dataclass
class DimBinder:
    """The name-to-size table for one function under trace."""

    variadic_rank: int = DEFAULT_VARIADIC_RANK
    sizes: dict[str, int] = field(default_factory=dict)
    variadics: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _next: int = 0

    def fresh(self) -> int:
        if self._next >= len(_PRIME_POOL):
            raise BindingError("ran out of distinct dimension primes")
        value = _PRIME_POOL[self._next]
        self._next += 1
        return value

    def bind(self, name: str) -> int:
        """Get the prime for a name, assigning one on first sight."""
        if name not in self.sizes:
            self.sizes[name] = self.fresh()
        return self.sizes[name]

    def bind_variadic(self, name: str | None) -> tuple[int, ...]:
        if name is None:
            return tuple(self.fresh() for _ in range(self.variadic_rank))
        if name not in self.variadics:
            self.variadics[name] = tuple(self.fresh() for _ in range(self.variadic_rank))
        return self.variadics[name]

    def describe(self, size: int) -> str:
        """Render a concrete size back as a name where one is known.

        A traced dimension of 10403 with `d_in`=101 and `d_out`=103 bound comes
        back as `d_in*d_out`, which is the whole point of using primes.
        """
        for name, value in self.sizes.items():
            if value == size:
                return name
        for name, values in self.variadics.items():
            if size in values:
                return f"{name}[{values.index(size)}]"
        factors = self._factor_names(size)
        if factors:
            return "*".join(factors)
        return str(size)

    def _factor_names(self, size: int) -> list[str]:
        by_value = {v: k for k, v in self.sizes.items()}
        for name, values in self.variadics.items():
            for index, value in enumerate(values):
                by_value.setdefault(value, f"{name}[{index}]")
        names: list[str] = []
        remaining = size
        for value, name in sorted(by_value.items(), reverse=True):
            while remaining % value == 0 and remaining > 1:
                names.append(name)
                remaining //= value
        return names if remaining == 1 and len(names) > 1 else []

    def render_shape(self, shape: tuple[int, ...]) -> str:
        return "(" + ", ".join(self.describe(s) for s in shape) + ")"


def _split_variadic(spec: ArraySpec) -> tuple[list[Dim], Dim | None, list[Dim]]:
    variadic_positions = [i for i, d in enumerate(spec.dims) if d.kind == "variadic"]
    if not variadic_positions:
        return list(spec.dims), None, []
    if len(variadic_positions) > 1:
        raise BindingError(f'more than one variadic dimension in "{spec}"')
    index = variadic_positions[0]
    return list(spec.dims[:index]), spec.dims[index], list(spec.dims[index + 1 :])


def _size_of(dim: Dim, binder: DimBinder) -> int:
    if dim.kind == "fixed":
        assert dim.size is not None
        return dim.size
    if dim.kind == "anonymous":
        return binder.fresh()
    if dim.kind == "named":
        assert dim.name is not None
        return binder.bind(dim.name)
    if dim.kind == "symbolic":
        return _eval_symbolic(dim, binder)
    raise BindingError(f"cannot size dimension {dim}")


def _eval_symbolic(dim: Dim, binder: DimBinder) -> int:
    assert dim.expr is not None
    import ast as _ast

    names = {
        node.id
        for node in _ast.walk(_ast.parse(dim.expr, mode="eval"))
        if isinstance(node, _ast.Name)
    }
    scope = {name: binder.bind(name) for name in names}
    try:
        value = eval(compile(_ast.parse(dim.expr, mode="eval"), "<dim>", "eval"), {}, scope)
    except Exception as exc:
        raise BindingError(f"could not evaluate dimension {dim.expr!r}: {exc}") from exc
    if not isinstance(value, int):
        raise BindingError(f"dimension {dim.expr!r} is not an integer")
    return value


def shape_for(spec: ArraySpec, binder: DimBinder) -> tuple[int, ...]:
    """Build the concrete shape an argument should be given."""
    prefix, variadic, suffix = _split_variadic(spec)
    shape: list[int] = [_size_of(d, binder) for d in prefix]
    if variadic is not None:
        shape.extend(binder.bind_variadic(variadic.name))
    shape.extend(_size_of(d, binder) for d in suffix)
    return tuple(shape)


def check_shape(spec: ArraySpec, shape: tuple[int, ...], binder: DimBinder) -> None:
    """Verify a traced shape against a spec, raising BindingError on mismatch.

    Names not yet in the binder get bound here, so a return-only name is allowed
    as long as it is used consistently.
    """
    prefix, variadic, suffix = _split_variadic(spec)
    fixed_rank = len(prefix) + len(suffix)

    if variadic is None:
        if len(shape) != fixed_rank:
            raise BindingError(
                f"expected {fixed_rank} dimensions, traced {len(shape)}",
                rule="rank-mismatch",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
                hint=_rank_hint(spec, shape, binder),
            )
        pairs = list(zip(spec.dims, shape))
    else:
        if len(shape) < fixed_rank:
            raise BindingError(
                f"expected at least {fixed_rank} dimensions, traced {len(shape)}",
                rule="rank-mismatch",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
            )
        middle = shape[len(prefix) : len(shape) - len(suffix)]
        _check_variadic(variadic, middle, binder, spec)
        pairs = list(zip(prefix, shape[: len(prefix)]))
        pairs += list(zip(suffix, shape[len(shape) - len(suffix) :]))

    for dim, size in pairs:
        _check_dim(dim, size, binder, spec, shape)


def _check_variadic(
    variadic: Dim, middle: tuple[int, ...], binder: DimBinder, spec: ArraySpec
) -> None:
    if variadic.name is None:
        return
    known = binder.variadics.get(variadic.name)
    if known is None:
        binder.variadics[variadic.name] = middle
    elif known != middle:
        raise BindingError(
            f"*{variadic.name} was {binder.render_shape(known)} earlier "
            f"but traced {binder.render_shape(middle)} here",
            rule="dim-inconsistent",
            expected=spec.shape_str(),
            got=binder.render_shape(middle),
        )


def _check_dim(
    dim: Dim, size: int, binder: DimBinder, spec: ArraySpec, shape: tuple[int, ...]
) -> None:
    if dim.kind == "anonymous":
        return
    if dim.broadcastable and size == 1:
        return

    if dim.kind == "fixed":
        if size != dim.size:
            raise BindingError(
                f"dimension is {binder.describe(size)}, annotated {dim.size}",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
            )
        return

    if dim.kind == "symbolic":
        expected = _eval_symbolic(dim, binder)
        if size != expected:
            raise BindingError(
                f"dimension {dim.expr} should be {expected}, traced {binder.describe(size)}",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
            )
        return

    assert dim.name is not None
    if dim.name in binder.sizes:
        expected = binder.sizes[dim.name]
        if size != expected:
            raise BindingError(
                f"`{dim.name}` is {expected} here, but the traced dimension is "
                f"{binder.describe(size)}",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
                hint=_swap_hint(dim.name, size, binder),
            )
    else:
        binder.sizes[dim.name] = size


def _swap_hint(name: str, size: int, binder: DimBinder) -> str:
    """When the traced size is another known dimension, say which one."""
    other = binder.describe(size)
    if other != str(size) and other != name:
        return f"this dimension is `{other}`, so the annotation likely names the wrong axis"
    return ""


def _rank_hint(spec: ArraySpec, shape: tuple[int, ...], binder: DimBinder) -> str:
    if len(shape) < len(spec.dims):
        merged = [binder.describe(s) for s in shape if "*" in binder.describe(s)]
        if merged:
            return f"{merged[0]} looks like two annotated axes flattened into one"
        return "the traced value has fewer axes than annotated, so something reduced"
    return "the traced value has more axes than annotated, so something added one"
