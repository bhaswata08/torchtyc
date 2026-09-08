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

import functools
import re
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


@functools.cache
def coprime_pool(scale: int) -> list[int]:
    """Sizes that are all multiples of `scale` and still factor apart.

    A trace normally hands each name a prime, which keeps distinct names
    distinct and lets a flattened axis be read back as the product it is. What
    a prime cannot do is divide, so a model that splits a width - `head_dim =
    d_model // n_heads` - gets a quotient that does not multiply back. Scaling
    every size by one shared factor repairs the split and keeps both
    properties: the primes are picked coprime to the factor, so a product of
    sizes still factors back into exactly the sizes that went into it.

    Small primes are used here rather than the pool above, because each size is
    multiplied by the factor and several sizes multiply together again into a
    tensor's element count, which has to stay inside int64.
    """
    if scale <= 1:
        return _PRIME_POOL
    found: list[int] = []
    candidate = 2
    while len(found) < len(_PRIME_POOL):
        if all(candidate % p for p in range(2, int(candidate**0.5) + 1)) and scale % candidate:
            found.append(candidate * scale)
        candidate += 1
    return found


def distant_prime(index: int) -> int:
    """A prime far from the ones a trace hands out, for a second probe.

    Re-building a model with a dimension moved only a little tells you nothing:
    `round(8 / 3 * d_model / 64) * 64` gives the same width for 101 and 103, so
    a size derived from `d_model` would look like a constant. Taking these from
    the far end of the pool moves the arithmetic far enough to show.
    """
    return _PRIME_POOL[-(index % len(_PRIME_POOL) + 1)]


def derived_name(names: tuple[str, ...]) -> str:
    """How a size computed by the user's `__init__` is written in a message.

    The number such a size comes out as is an artifact of the width torchtyc
    traced with, so printing it sends the reader looking for something their
    model does not contain. What it follows is the part that stays true.
    """
    return f"<from {' and '.join(names)}>"


def is_derived_name(rendered: str) -> bool:
    return rendered.startswith("<from ")


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
        suggestion: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.rule = rule
        self.expected = expected
        self.got = got
        self.hint = hint
        # A dim string the user could paste over the annotation, when the whole
        # traced shape has names. None when any axis cannot be named.
        self.suggestion = suggestion


@dataclass
class DimBinder:
    """The name-to-size table for one function under trace."""

    variadic_rank: int = DEFAULT_VARIADIC_RANK
    # A factor every size this binder issues is a multiple of. One means plain
    # primes, which is what an ordinary trace runs with. A second trace raises
    # it so that a width the model splits divides evenly. See `coprime_pool`.
    scale: int = 1
    # Take a constructor parameter's written default even when its name is one
    # of the annotated dimensions, rather than binding the name to a size of
    # this binder's own. Also only ever set on a second trace: `n_heads: int =
    # 8` names an axis and states a width in the same breath, and the width is
    # the half that has to hold for `d_model // n_heads` to come out whole.
    defaults_first: bool = False
    sizes: dict[str, int] = field(default_factory=dict)
    variadics: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # Sizes standing in for an unnamed `...`. They have no name the user wrote,
    # so reporting the prime would be noise; they render as `...` instead.
    anonymous: set[int] = field(default_factory=set)
    # Sizes standing in for a single unnamed `_` axis. Same reasoning as above,
    # but they render as `_` so the shape mirrors what the annotation wrote.
    anonymous_dims: set[int] = field(default_factory=set)
    # Sizes the user's own `__init__` computed, mapped to the dimensions they
    # follow. `d_ff = 8 / 3 * d_model` lands here as `(d_model,)`, and a literal
    # width written in the code lands as `()`. Filled in by a second probe, and
    # only on the failure path, where a size is about to be shown to someone.
    derived: dict[int, tuple[str, ...]] = field(default_factory=dict)
    # Fixed sizes written in annotations, such as the 3 in "b 3 h w". A traced
    # size that renders as its own digits is either such a literal, which is
    # written as itself and so is fit to suggest, or a size with no name, which
    # is not. Recording the literals tells the two apart.
    literals: set[int] = field(default_factory=set)
    _next: int = 0
    _anonymous_variadic: tuple[int, ...] | None = None

    def fresh(self) -> int:
        pool = coprime_pool(self.scale)
        if self._next >= len(pool):
            raise BindingError("ran out of distinct dimension primes")
        value = pool[self._next]
        self._next += 1
        return value

    def bind(self, name: str) -> int:
        """Get the prime for a name, assigning one on first sight."""
        if name not in self.sizes:
            self.sizes[name] = self.fresh()
        return self.sizes[name]

    def bind_variadic(self, name: str | None) -> tuple[int, ...]:
        if name is None:
            # Every bare `...` in one trace binds to the same axes. jaxtyping
            # treats each as independent; this is deliberately narrower, because
            # `f(x: Float[Tensor, "... d"], y: Float[Tensor, "... d"])` means one
            # batch shape to almost everyone who writes it, and tracing the two
            # with different shapes reports correct elementwise code as broken.
            if self._anonymous_variadic is None:
                sizes = tuple(self.fresh() for _ in range(self.variadic_rank))
                self.anonymous.update(sizes)
                self._anonymous_variadic = sizes
            return self._anonymous_variadic
        if name not in self.variadics:
            self.variadics[name] = tuple(self.fresh() for _ in range(self.variadic_rank))
        return self.variadics[name]

    @property
    def anonymous_variadic(self) -> tuple[int, ...] | None:
        """The axes every bare `...` in this trace shares, once one has bound."""
        return self._anonymous_variadic

    def bind_anonymous_variadic(self, sizes: tuple[int, ...]) -> None:
        """Record traced axes as the shared `...`, when nothing bound it first.

        A trace that only meets `...` in return positions has no argument to
        take the batch shape from, so the first return sets it and the rest are
        checked against that.

        These sizes came out of the trace rather than out of `fresh`, so they
        are not added to `anonymous`. A flattened `b*s` arriving here is still
        `b*s`, and relabelling it `...` would corrupt every later message about
        that size.
        """
        self._anonymous_variadic = sizes

    def bind_anonymous(self) -> int:
        value = self.fresh()
        self.anonymous_dims.add(value)
        return value

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
        if size in self.anonymous:
            return "..."
        if size in self.anonymous_dims:
            return "_"
        factors = self._factor_names(size)
        if factors:
            return "*".join(factors)
        if self.derived.get(size):
            return derived_name(self.derived[size])
        return str(size)

    def _factor_names(self, size: int) -> list[str]:
        # Only a size above one can be a factor. A name bound to 1 would never
        # shrink the remainder, and one bound to 0 would divide by zero.
        by_value = {v: k for k, v in self.sizes.items() if v > 1}
        for name, values in self.variadics.items():
            for index, value in enumerate(values):
                if value > 1:
                    by_value.setdefault(value, f"{name}[{index}]")
        # An anonymous axis has no name the user wrote, but its prime still
        # multiplies into a flattened size. Leaving it out here would let the
        # product survive into a message as a bare prime product.
        for value in self.anonymous:
            if value > 1:
                by_value.setdefault(value, "...")
        for value in self.anonymous_dims:
            if value > 1:
                by_value.setdefault(value, "_")

        # Binding order, not size order. A flatten of "b s" must read "b*s":
        # the hint names an axis order, so reversing it points at the wrong
        # outer axis. Fresh values come from a pool that grows monotonically,
        # so pool index is binding order and interleaves named, variadic and
        # anonymous axes issued between each other. Values bound directly to
        # traced sizes, such as return-only names, never come from the pool;
        # they were bound after, in sizes insertion order. Under a scale above
        # 1 each size is the scale times a prime coprime to it, the pool still
        # grows monotonically, and a true product still factors whichever order
        # is tried, so the same ordering holds there.
        pool = coprime_pool(self.scale)
        pool_index = {value: pos for pos, value in enumerate(pool)}
        size_pos = {name: pos for pos, name in enumerate(self.sizes)}
        variadic_pos = {name: pos for pos, name in enumerate(self.variadics)}

        def _key(item: tuple[int, str]) -> tuple[int, int]:
            value, name = item
            if value in pool_index:
                return (0, pool_index[value])
            if name in size_pos:
                return (1, size_pos[name])
            base = name.split("[")[0] if "[" in name else name
            if base in variadic_pos:
                return (1, len(size_pos) + variadic_pos[base])
            return (1, len(pool) + value)

        names: list[str] = []
        found = 0
        remaining = size
        for value, name in sorted(by_value.items(), key=_key):
            while remaining % value == 0 and remaining > 1:
                found += 1
                # Two anonymous axes flattened together are still one unnamed
                # run, exactly as `render_shape` collapses them.
                if not (name == "..." and names and names[-1] == "..."):
                    names.append(name)
                remaining //= value
        return names if remaining == 1 and found > 1 else []

    def is_flattened(self, size: int) -> bool:
        """Whether this size is a product of primes rather than one axis."""
        return size not in self.issued() and bool(self._factor_names(size))

    def issued(self) -> dict[int, str]:
        """Every size this binder handed out, mapped back to what it stands for."""
        table: dict[int, str] = {}
        for name, value in self.sizes.items():
            table.setdefault(value, name)
        for name, values in self.variadics.items():
            for index, value in enumerate(values):
                table.setdefault(value, f"{name}[{index}]")
        for value in self.anonymous:
            table.setdefault(value, "...")
        for value in self.anonymous_dims:
            table.setdefault(value, "_")
        return table

    def rename_primes(self, text: str) -> str:
        """Put axis names back into a message torch wrote in concrete sizes.

        A shape bug usually surfaces as a torch error quoting the sizes it saw,
        which are this binder's primes. Only a number the binder issued, or a
        product that factors wholly into such numbers, is replaced, so a rank,
        an index or a dtype width in the same sentence is left exactly as it
        was. A flattened axis is quoted as that product, and it leaks a prime
        just as plainly as a single axis does.
        """
        table = self.issued()
        if not table and not self.derived:
            return text

        def replace(match: re.Match[str]) -> str:
            value = int(match.group())
            if value in table:
                return table[value]
            factors = self._factor_names(value)
            if factors:
                return "*".join(factors)
            if self.derived.get(value):
                return derived_name(self.derived[value])
            return match.group()

        return re.sub(r"\d+", replace, text)

    def render_shape(self, shape: tuple[int, ...]) -> str:
        parts: list[str] = []
        for size in shape:
            rendered = self.describe(size)
            if rendered == "..." and parts and parts[-1] == "...":
                continue
            parts.append(rendered)
        return "(" + ", ".join(parts) + ")"

    def suggest_dims(self, shape: tuple[int, ...]) -> str | None:
        """Render a traced shape as a dim string fit to paste into an annotation.

        Returns None when any axis has no name the user would recognise, since
        suggesting `(107, out_features)` would be worse than suggesting nothing.
        """
        parts: list[str] = []
        for size in shape:
            # A flattened axis is not one axis, whatever its factors render as.
            if self.is_flattened(size):
                return None
            rendered = self.describe(size)
            if rendered == "...":
                if parts and parts[-1] == "...":
                    continue
                parts.append("...")
                continue
            # A product, a variadic index, an anonymous axis, or a size with no
            # name is not something to paste back. A fixed literal renders as
            # its own digits too, but it is written as itself, so it is fit to
            # suggest. is_flattened above already declines a true product, so
            # what renders as digits here is either a recorded literal or an
            # unbound size: a prime never bound, a derived width not yet
            # explained, or a partial product with a literal such as 303 from
            # 101 * 3. Only the first is pasteable.
            if (
                "*" in rendered
                or "[" in rendered
                or rendered == "_"
                or (rendered == str(size) and size not in self.literals)
                or is_derived_name(rendered)
            ):
                return None
            parts.append(rendered)
        # jaxtyping allows at most one variadic, and so does the binder. Two
        # separate anonymous runs cannot be written as a valid dim string.
        if parts.count("...") > 1:
            return None
        return " ".join(parts) if parts else None


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
        # Remember the literal so suggest_dims can tell it apart from an
        # unbound size. Both render as their own digits, but only the literal
        # is written as itself and so is fit to paste back.
        binder.literals.add(dim.size)
        return dim.size
    if dim.kind == "anonymous":
        return binder.bind_anonymous()
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
                suggestion=binder.suggest_dims(shape),
            )
        pairs = list(zip(spec.dims, shape))
    else:
        if len(shape) < fixed_rank:
            raise BindingError(
                f"expected at least {fixed_rank} dimensions, traced {len(shape)}",
                rule="rank-mismatch",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
                suggestion=binder.suggest_dims(shape),
            )
        middle = shape[len(prefix) : len(shape) - len(suffix)]
        _check_variadic(variadic, middle, binder, spec)
        pairs = list(zip(prefix, shape[: len(prefix)]))
        pairs += list(zip(suffix, shape[len(shape) - len(suffix) :]))

    for dim, size in pairs:
        _check_dim(dim, size, binder, spec, shape)


def _render_batch(sizes: tuple[int, ...], known: tuple[int, ...], binder: DimBinder) -> str:
    """Render batch axes by their position in the shared `...`.

    `render_shape` collapses a run of anonymous axes into one `...`, which is
    right for a shape a reader is meant to skim but useless here: both sides of
    a batch mismatch would read `(...)`. Indexing each axis instead shows which
    position went missing or moved.
    """
    parts = [f"...[{known.index(s)}]" if s in known else binder.describe(s) for s in sizes]
    return f"({', '.join(parts)})"


def _check_variadic(
    variadic: Dim, middle: tuple[int, ...], binder: DimBinder, spec: ArraySpec
) -> None:
    if variadic.name is None:
        # Every bare `...` in one trace shares its axes, so the return's middle
        # is checked against that binding exactly as a named variadic is.
        # Skipping it would let a return that drops or reorders a batch axis
        # pass, which is the mistake `... d` is usually written to catch.
        known = binder.anonymous_variadic
        if known is None:
            # First sight binds it, exactly as a named variadic binds on first
            # sight. Returning without recording would leave two returns in one
            # trace unchecked against each other.
            binder.bind_anonymous_variadic(middle)
            return
        if known == middle:
            return
        raise BindingError(
            f"the batch `...` traced {_render_batch(middle, known, binder)} here "
            f"but {_render_batch(known, known, binder)} earlier",
            rule="dim-inconsistent",
            expected=spec.shape_str(),
            got=_render_batch(middle, known, binder),
        )
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
        assert dim.size is not None
        binder.literals.add(dim.size)
        if size != dim.size:
            raise BindingError(
                f"dimension is {binder.describe(size)}, annotated {dim.size}",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
            )
        return

    if dim.kind == "symbolic":
        # The comparison runs over the binder's primes, but their sum or
        # product is not a size any reader can place, so the message names the
        # expression and the axis instead of quoting the arithmetic.
        if size != _eval_symbolic(dim, binder):
            traced = binder.describe(size)
            named = traced != str(size)
            raise BindingError(
                f"annotated `{dim.expr}`, but the traced dimension is `{traced}`"
                if named
                else f"annotated `{dim.expr}`, but this axis traced {size}",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
            )
        return

    assert dim.name is not None
    if dim.name in binder.sizes:
        expected = binder.sizes[dim.name]
        if size != expected:
            traced = binder.describe(size)
            named = traced != str(size)
            raise BindingError(
                f"annotated `{dim.name}`, but the traced dimension is `{traced}`"
                if named
                else f"annotated `{dim.name}`, but this axis traced {size}",
                expected=spec.shape_str(),
                got=binder.render_shape(shape),
                hint=_swap_hint(dim.name, size, binder),
                suggestion=binder.suggest_dims(shape),
            )
    else:
        binder.sizes[dim.name] = size


def _swap_hint(name: str, size: int, binder: DimBinder) -> str:
    """When the traced size is another known dimension, say which one."""
    if binder.is_flattened(size):
        # Two axes multiplied into one is not a misnamed axis, and `_rank_hint`
        # already says so in the words that fit it.
        return ""
    other = binder.describe(size)
    if other in ("_", "...") or is_derived_name(other):
        # A width the model computed is not an axis the annotation could have
        # meant instead, so there is no swap to point at. The note beside the
        # message already says where the width came from.
        return ""
    if other != str(size) and other != name:
        return f"this dimension is `{other}`, so the annotation likely names the wrong axis"
    return ""


def _rank_hint(spec: ArraySpec, shape: tuple[int, ...], binder: DimBinder) -> str:
    if len(shape) < len(spec.dims):
        merged = [binder.describe(s) for s in shape if binder.is_flattened(s)]
        if merged:
            return f"{merged[0]} looks like two annotated axes flattened into one"
        return "the traced value has fewer axes than annotated, so something reduced"
    return "the traced value has more axes than annotated, so something added one"
