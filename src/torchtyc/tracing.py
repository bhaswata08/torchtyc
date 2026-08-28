"""The trace: build meta tensors, call the function, read the shapes back.

This module is the only one that imports torch, and it always runs inside the
worker subprocess against the *project's* interpreter, never the one hosting
the language server.

Nothing here allocates memory. `torch.device("meta")` gives tensors that carry
shape, dtype, and stride but own no storage, so calling a model under it costs
roughly a dictionary lookup per operator and zero FLOPs.
"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import Any

import torch

from .annotations import ArraySpec, OpaqueSpec, Spec, TupleSpec
from .binding import BindingError, DimBinder, check_shape, shape_for
from .discovery import ClassInfo, Param, Target

# jaxtyping dtype name -> (dtype used to build an argument, dtypes accepted on
# the way out). Building picks one representative; checking accepts the family.
_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_COMPLEX = (torch.complex64, torch.complex128)
_SIGNED = (torch.int8, torch.int16, torch.int32, torch.int64)
_UNSIGNED = (torch.uint8,) + tuple(
    getattr(torch, name) for name in ("uint16", "uint32", "uint64") if hasattr(torch, name)
)

DTYPES: dict[str, tuple[torch.dtype, tuple[torch.dtype, ...]]] = {
    "Float": (torch.float32, _FLOATS),
    "Complex": (torch.complex64, _COMPLEX),
    "Inexact": (torch.float32, _FLOATS + _COMPLEX),
    "Int": (torch.int64, _SIGNED),
    "Integer": (torch.int64, _SIGNED + _UNSIGNED),
    "UInt": (torch.uint8, _UNSIGNED),
    "Bool": (torch.bool, (torch.bool,)),
    "Real": (torch.float32, _FLOATS + _SIGNED + _UNSIGNED),
    "Num": (torch.float32, _FLOATS + _COMPLEX + _SIGNED + _UNSIGNED),
    "Shaped": (torch.float32, ()),  # () means "accept anything"
    "Key": (torch.int64, _SIGNED + _UNSIGNED),
    "Float16": (torch.float16, (torch.float16,)),
    "BFloat16": (torch.bfloat16, (torch.bfloat16,)),
    "Float32": (torch.float32, (torch.float32,)),
    "Float64": (torch.float64, (torch.float64,)),
    "Complex64": (torch.complex64, (torch.complex64,)),
    "Complex128": (torch.complex128, (torch.complex128,)),
    "Int8": (torch.int8, (torch.int8,)),
    "Int16": (torch.int16, (torch.int16,)),
    "Int32": (torch.int32, (torch.int32,)),
    "Int64": (torch.int64, (torch.int64,)),
    "UInt8": (torch.uint8, (torch.uint8,)),
}


class TraceSkipped(Exception):
    """The target could not be set up. Carries a rule name and a message."""

    def __init__(self, rule: str, message: str, hint: str = ""):
        super().__init__(message)
        self.rule = rule
        self.message = message
        self.hint = hint


@dataclass
class TraceResult:
    binder: DimBinder
    returned: Any
    arguments: dict[str, Any]
    # Shapes of every annotated argument, for hover and inlay hints.
    argument_shapes: dict[str, tuple[int, ...]]


def build_dtype(spec: ArraySpec) -> torch.dtype:
    entry = DTYPES.get(spec.dtype)
    if entry is None:
        raise TraceSkipped("unsupported-annotation", f"unknown dtype `{spec.dtype}`")
    return entry[0]


def accepted_dtypes(spec: ArraySpec) -> tuple[torch.dtype, ...]:
    entry = DTYPES.get(spec.dtype)
    return entry[1] if entry else ()


def build_tensor(spec: ArraySpec, binder: DimBinder) -> torch.Tensor:
    shape = shape_for(spec, binder)
    tensor = torch.empty(shape, dtype=build_dtype(spec), device="meta")
    if spec.array_type.split(".")[-1] == "Parameter":
        return torch.nn.Parameter(tensor, requires_grad=False)
    return tensor


def build_value(param: Param, binder: DimBinder, dim_names: set[str]) -> Any:
    """Produce an argument for one parameter.

    The interesting case is a plain `int` whose name matches a dimension name in
    the annotations, as in `Linear(in_features, out_features)` feeding a forward
    annotated `"... in_features"`. Passing the dimension's prime is what lets a
    module be constructed without the user writing a separate spec.
    """
    if isinstance(param.spec, ArraySpec):
        return build_tensor(param.spec, binder)

    if isinstance(param.spec, TupleSpec):
        return tuple(
            build_tensor(item, binder) for item in param.spec.items if isinstance(item, ArraySpec)
        )

    # Only an integer parameter can stand for a dimension. Without this guard a
    # constructor parameter called `device` would be handed a prime whenever a
    # dimension happened to share its name.
    if param.name in dim_names and param.plain_type in (None, "int"):
        return binder.bind(param.name)

    plain = param.plain_type
    if plain == "int":
        # Record it under its own name even though no annotation mentions it.
        # `Linear(in_features, out_features)` with a forward that only names
        # `in_features` still lets a message say "this dimension is
        # out_features" rather than "this dimension is 103".
        return binder.bind(param.name)
    if plain == "float":
        return 1.0
    if plain == "bool":
        return False
    if plain == "str":
        return ""
    if plain in ("torch.device", "device"):
        return torch.device("meta")
    if plain in ("torch.dtype", "dtype"):
        return torch.float32
    if plain == "Tensor" or (plain or "").endswith(".Tensor"):
        # An unannotated tensor: one dimension is the least constraining guess.
        return torch.empty((binder.fresh(),), device="meta")

    if param.has_default:
        raise _UseDefault()

    raise TraceSkipped(
        "unresolved-arg",
        f"cannot build a value for `{param.name}`"
        + (f" of type `{plain}`" if plain else " because it has no annotation"),
        hint="give it a default, or annotate it with a jaxtyping array type",
    )


class _UseDefault(Exception):
    """Internal: the parameter has a default, so leave it out of the call."""


def instantiate(owner: ClassInfo, cls: type, binder: DimBinder, dim_names: set[str]) -> Any:
    """Construct a module on the meta device so its parameters cost nothing."""
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    positional_open = True
    for param in owner.init_params:
        if param.name in ("args", "kwargs"):
            continue
        if param.positional_only and not positional_open:
            continue
        try:
            value = build_value(param, binder, dim_names)
        except _UseDefault:
            if param.positional_only:
                positional_open = False
            continue
        except TraceSkipped as exc:
            raise TraceSkipped(
                "uninstantiable",
                f"cannot construct `{owner.name}`: {exc.message}",
                hint=exc.hint,
            ) from exc
        if param.positional_only:
            args.append(value)
        else:
            kwargs[param.name] = value

    with torch.device("meta"), _quiet_init():
        try:
            return cls(*args, **kwargs)
        except TypeError as exc:
            raise TraceSkipped("uninstantiable", f"cannot construct `{owner.name}`: {exc}") from exc


@contextlib.contextmanager
def _quiet_init():
    """Neutralise in-place initialisers that reject meta tensors.

    Most of `torch.nn.init` works on meta tensors because it only records
    shapes, but a few (anything reading a value back, such as `orthogonal_`)
    raise. Since initial values never affect a shape trace, replacing them with
    the identity is safe and removes a class of spurious failures.
    """
    from torch.nn import init

    names = [
        "orthogonal_",
        "sparse_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "trunc_normal_",
        "normal_",
        "uniform_",
    ]
    saved = {}
    for name in names:
        original = getattr(init, name, None)
        if original is None:
            continue
        saved[name] = original
        setattr(init, name, _identity_init(original))
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(init, name, original)


def _identity_init(original):
    def wrapper(tensor, *args, **kwargs):
        if getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
            return tensor
        return original(tensor, *args, **kwargs)

    wrapper.__name__ = getattr(original, "__name__", "init")
    return wrapper


def resolve_callable(module: Any, target: Target, binder: DimBinder) -> tuple[Any, list[Param]]:
    """Find the function to call, constructing an instance when it is a method."""
    params = list(target.params)

    if target.owner is None:
        fn = getattr(module, target.name, None)
        if fn is None:
            raise TraceSkipped("trace-error", f"`{target.name}` is not defined after import")
        return fn, params

    cls = getattr(module, target.owner.name, None)
    if cls is None:
        raise TraceSkipped("trace-error", f"`{target.owner.name}` is not defined after import")

    if "staticmethod" in target.decorators:
        return getattr(cls, target.name), params
    if "classmethod" in target.decorators:
        return getattr(cls, target.name), params[1:]
    if "property" in target.decorators:
        raise TraceSkipped("uninstantiable", "properties are not traced")

    instance = instantiate(target.owner, cls, binder, target.dim_names)
    return getattr(instance, target.name), params[1:]  # drop self


def trace(module: Any, target: Target, variadic_rank: int) -> TraceResult:
    """Call one target on meta tensors and hand back what came out."""
    binder = DimBinder(variadic_rank=variadic_rank)
    fn, params = resolve_callable(module, target, binder)

    arguments: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    # A parameter declared before `/` cannot be passed by name, so it goes in a
    # separate list. Once one of them falls back to its default every later
    # positional-only parameter must too, since defaults are trailing.
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    positional_open = True
    for param in params:
        if param.positional_only and not positional_open:
            continue
        try:
            value = build_value(param, binder, target.dim_names)
        except _UseDefault:
            if param.positional_only:
                positional_open = False
            continue
        arguments[param.name] = value
        if param.positional_only:
            positional.append(value)
        else:
            keywords[param.name] = value
        if isinstance(value, torch.Tensor):
            shapes[param.name] = tuple(value.shape)

    # nn.Module.__call__ runs hooks that can allocate; calling forward directly
    # keeps the trace to the user's own code.
    with torch.device("meta"), torch.no_grad():
        returned = fn(*positional, **keywords)

    return TraceResult(
        binder=binder, returned=returned, arguments=arguments, argument_shapes=shapes
    )


def describe(value: Any, binder: DimBinder) -> str:
    if isinstance(value, torch.Tensor):
        return f"{_dtype_name(value.dtype)}[{binder.render_shape(tuple(value.shape))}]"
    if isinstance(value, tuple):
        return "(" + ", ".join(describe(v, binder) for v in value) + ")"
    return type(value).__name__


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def check_return(spec: Spec | None, value: Any, binder: DimBinder) -> list[dict[str, str]]:
    """Compare a returned value against its annotation.

    Returns a list of problem dicts rather than raising, so one function can
    report a mismatch in every element of a tuple return at once.
    """
    problems: list[dict[str, str]] = []

    if spec is None or isinstance(spec, OpaqueSpec):
        return problems

    if isinstance(spec, TupleSpec):
        if not isinstance(value, (tuple, list)):
            return [
                {
                    "rule": "tuple-arity",
                    "message": f"annotated as a tuple but returned {type(value).__name__}",
                    "expected": str(spec),
                    "got": describe(value, binder),
                }
            ]
        if len(value) != len(spec.items):
            return [
                {
                    "rule": "tuple-arity",
                    "message": f"annotated {len(spec.items)} values, returned {len(value)}",
                    "expected": str(spec),
                    "got": describe(value, binder),
                }
            ]
        for index, (item, element) in enumerate(zip(spec.items, value)):
            for problem in check_return(item, element, binder):
                problem["message"] = f"element {index}: {problem['message']}"
                problems.append(problem)
        return problems

    assert isinstance(spec, ArraySpec)

    if not isinstance(value, torch.Tensor):
        return [
            {
                "rule": "not-a-tensor",
                "message": f"annotated as a tensor but returned {type(value).__name__}",
                "expected": str(spec),
                "got": describe(value, binder),
            }
        ]

    accepted = accepted_dtypes(spec)
    if accepted and value.dtype not in accepted:
        problems.append(
            {
                "rule": "dtype-mismatch",
                "message": f"returned {_dtype_name(value.dtype)}, annotated `{spec.dtype}`",
                "expected": spec.dtype,
                "got": _dtype_name(value.dtype),
            }
        )

    if value.device.type != "meta":
        problems.append(
            {
                "rule": "device-mismatch",
                "message": f"the returned tensor is on {value.device}, not the traced meta device",
                "expected": "meta",
                "got": str(value.device),
                "hint": "a hard-coded .cuda() or .to(device) makes the function untraceable",
            }
        )

    try:
        check_shape(spec, tuple(value.shape), binder)
    except BindingError as exc:
        problems.append(
            {
                "rule": exc.rule,
                "message": exc.message,
                "expected": exc.expected or spec.shape_str(),
                "got": exc.got or binder.render_shape(tuple(value.shape)),
                "hint": exc.hint,
            }
        )

    return problems


def signature_defaults(fn: Any) -> set[str]:
    try:
        return {
            name
            for name, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
    except (TypeError, ValueError):
        return set()
