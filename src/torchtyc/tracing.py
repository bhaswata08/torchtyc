"""The trace: build meta tensors, call the function, read the shapes back.

This module is the only one that imports torch, and it always runs inside the
worker subprocess against the *project's* interpreter, never the one hosting
the language server.

Nothing here allocates memory. `torch.device("meta")` gives tensors that carry
shape, dtype, and stride but own no storage, so calling a model under it costs
roughly a dictionary lookup per operator and zero FLOPs.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .annotations import ArraySpec, OpaqueSpec, Spec, TupleSpec
from .binding import BindingError, DimBinder, check_shape, shape_for
from .discovery import ClassInfo, InitDef, Param, Target

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
        # The axis carries no name the user wrote, so it binds as anonymous and
        # renders as `_` instead of leaking the synthetic prime.
        return torch.empty((binder.bind_anonymous(),), device="meta")

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


class TraceFailed(Exception):
    """The traced call raised.

    Carries the binder alongside the original error, because torch reports a
    shape bug with the concrete sizes it saw, and those are torchtyc's primes.
    The caller needs the binder to put the user's own axis names back.
    """

    def __init__(self, error: BaseException, binder: DimBinder):
        super().__init__(str(error))
        self.error = error
        self.binder = binder


def live_init(owner: ClassInfo, cls: type) -> InitDef | None:
    """Which of the `__init__` definitions in the body this import produced.

    A class that writes `__init__` in both arms of a guard has two definitions
    and one constructor. Source order cannot say which arm ran, but the code
    object behind the live `__init__` names the lines it was written at.
    """
    if len(owner.inits) < 2:
        return owner.init
    code = getattr(getattr(cls, "__init__", None), "__code__", None)
    if code is None:
        return owner.init
    line = code.co_firstlineno - 1
    for candidate in owner.inits:
        if candidate.def_line <= line <= candidate.end_line:
            return candidate
    return owner.init


def instantiate(owner: ClassInfo, cls: type, binder: DimBinder, dim_names: set[str]) -> Any:
    """Construct a module on the meta device so its parameters cost nothing."""
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    positional_open = True
    init = live_init(owner, cls)
    for param in init.params if init else []:
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
                f"cannot construct `{owner.qualname or owner.name}`: {exc.message}",
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
            raise TraceSkipped(
                "uninstantiable", f"cannot construct `{owner.qualname or owner.name}`: {exc}"
            ) from exc


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


class NotLive(Exception):
    """A guarded definition that this import did not actually produce.

    Scanning descends into every branch, so a `class` under
    `if sys.version_info >= (3, 99)` is discovered even though the import never
    ran it, and a name defined in both branches of a guard is discovered twice.
    Neither is a finding about the user's code, so the caller drops the target
    without reporting anything.
    """


def _code_positions(obj: Any) -> list[tuple[str, int]]:
    """Where the code behind an object was written: (filename, 1-based line)."""
    code = getattr(obj, "__code__", None)
    if code is not None:
        return [(code.co_filename, code.co_firstlineno)]
    if isinstance(obj, type):
        found = []
        for value in vars(obj).values():
            fn = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
            code = getattr(fn, "__code__", None)
            if code is not None:
                found.append((code.co_filename, code.co_firstlineno))
        return found
    return []


def is_live_definition(obj: Any, module: Any, first_line: int, last_line: int) -> bool:
    """Whether `obj` is what this particular definition site produced.

    Two things disqualify it: coming from another module altogether, which is
    what a `try: from fast import Block / except ImportError: class Block`
    fallback leaves behind, and being written at other lines of this same file,
    which is what the losing branch of a guard sees. An object whose code cannot
    be placed at all counts as live, because a decorator returning a wrapper
    from some other file is not evidence of shadowing.
    """
    module_name = getattr(module, "__name__", None)
    if module_name is not None and getattr(obj, "__module__", module_name) != module_name:
        return False

    path = getattr(module, "__file__", None)
    if path is None:
        return True
    here = Path(path).resolve()
    same_file = [
        line for filename, line in _code_positions(obj) if Path(filename).resolve() == here
    ]
    if not same_file:
        return True
    return any(first_line <= line - 1 <= last_line for line in same_file)


def resolve_qualname(
    module: Any,
    qualname: str,
    *,
    conditional: bool = False,
    span: tuple[int, int] = (0, 0),
) -> Any:
    """Walk a dotted qualname from the module down to the object it names.

    Nesting means the name is no longer a module attribute: `Outer.Inner` needs
    two lookups. A qualname holding `<locals>` names something built inside a
    function call, which no amount of `getattr` can reach, so it is skipped with
    a diagnostic rather than left silently unchecked.

    A `conditional` definition sits under a branch that the import may not have
    taken, so it has to prove it is the live one. An unguarded definition that
    is missing after import stays an error, because there the absence is real.
    """
    if "<locals>" in qualname:
        raise TraceSkipped(
            "local-definition",
            f"`{qualname}` is defined inside a function body, so it cannot be checked",
            hint="move it to module level or into a class to bring it under the checker",
        )
    found = module
    for part in qualname.split("."):
        found = getattr(found, part, None)
        if found is None:
            if conditional:
                raise NotLive(qualname)
            raise TraceSkipped("trace-error", f"`{qualname}` is not defined after import")
    if conditional and not is_live_definition(found, module, span[0], span[1]):
        raise NotLive(qualname)
    return found


def resolve_callable(module: Any, target: Target, binder: DimBinder) -> tuple[Any, list[Param]]:
    """Find the function to call, constructing an instance when it is a method."""
    params = list(target.params)

    if target.owner is None:
        return resolve_qualname(
            module,
            target.qualname,
            conditional=target.conditional,
            span=(target.def_line, target.end_line),
        ), params

    owner = target.owner
    cls = resolve_qualname(
        module,
        owner.qualname,
        conditional=owner.conditional,
        span=(owner.def_line, owner.end_line),
    )

    if "property" in target.decorators:
        raise TraceSkipped("uninstantiable", "properties are not traced")

    if "staticmethod" in target.decorators:
        return _live_method(module, cls, target), params
    if "classmethod" in target.decorators:
        return _live_method(module, cls, target), params[1:]

    instance = instantiate(owner, cls, binder, owner.dim_names)
    return _live_method(module, instance, target), params[1:]  # drop self


def _live_method(module: Any, owner: Any, target: Target) -> Any:
    """The bound method for a target, once it is known to be this definition.

    A guarded `def` in a live class body is the same problem as a guarded one at
    module level: the class exists, so the owner resolves, but the method behind
    the name may be a base class's or the winner of the other branch. The class
    boundary cannot see that, so the check belongs here, where the callable the
    tracer will actually run is finally in hand.
    """
    found = getattr(owner, target.name, None)
    if found is None:
        if target.conditional:
            raise NotLive(target.qualname)
        raise TraceSkipped("trace-error", f"`{target.qualname}` is not defined after import")
    if target.conditional and not is_live_definition(
        found, module, target.def_line, target.end_line
    ):
        raise NotLive(target.qualname)
    return found


def trace(module: Any, target: Target, variadic_rank: int) -> TraceResult:
    """Call one target on meta tensors and hand back what came out.

    Every way this can fail on the user's own code - constructing the module,
    building an argument, running the body - comes back as `TraceFailed`, which
    carries the binder. That is what lets the caller report the failure in the
    user's axis names instead of the primes torch actually saw.
    """
    binder = DimBinder(variadic_rank=variadic_rank)
    try:
        return _trace(module, target, binder)
    except (TraceSkipped, NotLive, TraceFailed):
        raise
    except Exception as exc:
        raise TraceFailed(exc, binder) from exc


def _trace(module: Any, target: Target, binder: DimBinder) -> TraceResult:
    fn, params = resolve_callable(module, target, binder)

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
        if inspect.iscoroutine(returned):
            returned = _settle_coroutine(returned, awaited=target.is_async)

    return TraceResult(binder=binder, returned=returned, argument_shapes=shapes)


def _settle_coroutine(coroutine: Any, *, awaited: bool) -> Any:
    """Finish with a coroutine, either by running it or by putting it down.

    An `async def` target returns a coroutine because that is what calling one
    does, so running it to completion is what gives the shape check a tensor to
    look at. A plain `def` returning one is a forgotten await, and repairing it
    here would hide the bug: the coroutine comes back untouched for
    `check_return` to report as `not-a-tensor`.

    Either way it gets closed. A coroutine that is never started raises
    `RuntimeWarning: coroutine was never awaited` when it is collected, and the
    worker's stderr is not the place for that. Closing one that already
    finished, however it finished, does nothing.
    """
    if not awaited:
        coroutine.close()
        return coroutine
    try:
        return asyncio.run(coroutine)
    finally:
        coroutine.close()


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
        suggested = _dtype_class(value.dtype)
        problems.append(
            {
                "rule": "dtype-mismatch",
                "message": f"returned {_dtype_name(value.dtype)}, annotated `{spec.dtype}`",
                "expected": spec.dtype,
                "got": _dtype_name(value.dtype),
                "suggestion": (
                    f'{suggested}[{spec.array_type}, "{" ".join(str(d) for d in spec.dims)}"]'
                    if suggested
                    else None
                ),
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
                "suggestion": (
                    f'{spec.dtype}[{spec.array_type}, "{exc.suggestion}"]'
                    if exc.suggestion
                    else None
                ),
            }
        )

    return problems


def _dtype_class(dtype: torch.dtype) -> str | None:
    """The broadest jaxtyping dtype class that accepts this dtype.

    Preferring the broad class (`Float` over `Float32`) matches how annotations
    are normally written, and keeps the suggestion from over-constraining a
    function that is fine in half precision too.
    """
    for name in ("Float", "Complex", "Int", "UInt", "Bool"):
        entry = DTYPES.get(name)
        if entry and dtype in entry[1]:
            return name
    return None
