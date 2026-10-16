"""Run a user's real entry point with every tensor on the meta device.

`torchtyc run` imports a script and calls its entry point inside `install()`.
Meta tensors carry shape, dtype and stride but own no storage, so a large
model builds and runs a forward pass in well under a second, and a shape bug
raises at the user's line just as it would on a GPU.

Like tracing.py this module imports torch, and only the worker loads it.
Every patch `install()` applies is recorded and taken back out on exit.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from .effects import unwrap_blocked

META = torch.device("meta")


class MetaValueRead(BaseException):
    """The run needed a tensor's value, which a meta tensor does not have.

    A BaseException, so a user's `except Exception:` cannot swallow the stop
    and carry on down a branch the real run would never take.
    """

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


class Unsupported(BaseException):
    """The run reached something the meta runtime cannot stand in for."""

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


# Ops whose output depends on tensor values rather than shapes. Every other op
# computes its output shape from input shapes alone, which is all meta has.
_VALUE_OPS = frozenset(
    {
        "aten::_local_scalar_dense",  # .item(), bool(t), int(t), x[:t]
        "aten::nonzero",  # torch.where(cond), argwhere
        "aten::masked_select",
        "aten::_unique",
        "aten::_unique2",
        "aten::unique_dim",
        "aten::unique_consecutive",
        "aten::bincount",
        "aten::equal",
    }
)
# A tensor of repeats sizes the output by its values; an int does not.
_REPEATS_BY_VALUE = frozenset({"Tensor", "self_Tensor"})
_MASK_DTYPES = (torch.bool, torch.uint8)
# Value reads that C++ makes without dispatching an op, known by their message.
_META_VALUE_MESSAGES = (
    "cannot be called on meta tensors",
    "unallocated storage",
    "Cannot copy out of meta tensor",
)
_NO_CUDA_MESSAGES = (
    "Found no NVIDIA driver",
    "Torch not compiled with CUDA enabled",
    "no CUDA GPUs are available",
    "CUDA driver initialization failed",
)
# Tensor methods that hand a value to Python. They all pass through
# __torch_function__, so the function mode sees them before C++ does.
_VALUE_METHODS = frozenset(
    {torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.numpy, torch.Tensor.__array__}
)
_DEVICE_MOVES = frozenset({torch.Tensor.cuda, torch.Tensor.cpu})


def is_meta_value_error(exc: BaseException) -> bool:
    return isinstance(exc, (RuntimeError, NotImplementedError, TypeError)) and any(
        message in str(exc) for message in _META_VALUE_MESSAGES
    )


def from_loader(exc: BaseException) -> bool:
    """Whether the exception, or one it was raised from, left a patched loader."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if getattr(current, "_torchtyc_loader", False):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


@dataclass
class RunState:
    """What one run has learned, read back by the worker for its report."""

    path_map: tuple[tuple[str, str], ...] = ()
    active: list[str] = field(default_factory=list)
    inactive: list[str] = field(default_factory=list)
    ignored: set[str] = field(default_factory=set)
    # Integer fields of the most recently loaded model's config.
    model_config: dict[str, int] = field(default_factory=dict)
    # einops axis -> every size it was bound to during the run.
    einops_axes: dict[str, set[int]] = field(default_factory=dict)

    def remap(self, path: Any) -> Any:
        if not self.path_map or not isinstance(path, (str, bytes, os.PathLike)):
            return path
        text = os.fsdecode(path)
        for old, new in self.path_map:
            if text == old or text.startswith(old.rstrip("/") + "/"):
                mapped = new + text[len(old) :]
                return type(path)(mapped) if isinstance(path, PurePath) else mapped
        return path

    @contextlib.contextmanager
    def loading(self) -> Iterator[None]:
        """Mark errors from inside a patched loader.

        A missing file or a blocked download there is an input problem, not a
        bug in the script, and `classify` needs to tell the two apart.
        """
        try:
            yield
        except (MetaValueRead, Unsupported, KeyboardInterrupt):
            raise
        except BaseException as exc:
            with contextlib.suppress(AttributeError, TypeError):
                exc._torchtyc_loader = True
            raise


STATE: RunState | None = None


def _state() -> RunState:
    if STATE is None:
        raise RuntimeError("the meta runtime is not installed")
    return STATE


class _Patches:
    """Attribute replacements, undone in reverse order."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, bool, Any]] = []

    def set(self, owner: Any, name: str, value: Any) -> None:
        own = vars(owner)
        self._undo.append((owner, name, name in own, own.get(name)))
        setattr(owner, name, value)

    def restore(self) -> None:
        while self._undo:
            owner, name, had, old = self._undo.pop()
            if had:
                setattr(owner, name, old)
            else:
                delattr(owner, name)


def _is_meta(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.is_meta


def _to_meta(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and not value.is_meta:
        return value.to(META)
    return value


def _device_like(value: Any) -> bool:
    return isinstance(value, (str, torch.device)) or type(value) is int


def _has_mask(indices: Any) -> bool:
    return any(
        isinstance(index, torch.Tensor) and index.dtype in _MASK_DTYPES for index in indices or ()
    )


def _reads_values(func: Any, name: str, args: tuple, kwargs: dict) -> bool:
    if name in _VALUE_OPS:
        return True
    if name == "aten::repeat_interleave":
        return func._overloadname in _REPEATS_BY_VALUE and kwargs.get("output_size") is None
    if name == "aten::index":
        return len(args) > 1 and _has_mask(args[1])
    return False


def _short(name: str) -> str:
    return f"`{name.removeprefix('aten::')}`"


class _DeviceMode(TorchFunctionMode):
    """Send every tensor to meta, and stop at Python-level value reads."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if args and _is_meta(args[0]):
            if func in _VALUE_METHODS:
                raise MetaValueRead(f"`.{func.__name__}()`")
            if func is torch.Tensor.__format__:
                spec = args[1] if len(args) > 1 else kwargs.get("format_spec", "")
                if spec:
                    raise MetaValueRead("formatting a tensor's value")
        if func in _DEVICE_MOVES:
            return args[0].to(META)
        if func is torch.Tensor.pin_memory:
            return args[0]
        if func is torch.Tensor.to:
            args = (args[0], *(META if _device_like(a) else a for a in args[1:]))
        if kwargs.get("device") is not None:
            kwargs["device"] = META
        return func(*args, **kwargs)


class _MetaDispatchMode(TorchDispatchMode):
    """Stop at value-reading ops, and move stray real tensors onto meta."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name
        if _reads_values(func, name, args, kwargs):
            raise MetaValueRead(_short(name))
        if name.startswith("aten::index_put") and len(args) > 1 and _has_mask(args[1]):
            # Assigning through a mask keeps the target's shape, which is all
            # the run needs; which elements it touches depends on values.
            return args[0] if name.endswith("_") else args[0].clone()
        if kwargs.get("device") is not None:
            kwargs = {**kwargs, "device": META}
        args, kwargs = tree_map(_to_meta, (args, kwargs))
        try:
            return func(*args, **kwargs)
        except NotImplementedError as exc:
            if is_meta_value_error(exc):
                raise MetaValueRead(_short(name)) from exc
            raise Unsupported(f"torch has no meta kernel for {_short(name)}") from exc


@contextlib.contextmanager
def install(path_map: Sequence[Sequence[str]] = (), *, hf_flags: bool = True) -> Iterator[RunState]:
    """Put the meta runtime in place for the length of the block.

    `hf_flags=False` leaves transformers' `is_tracing` and
    `is_torchdynamo_compiling` alone; only the canary test uses it, to prove
    those two patches still matter.
    """
    global STATE
    if STATE is not None:
        raise RuntimeError("the meta runtime is already installed")
    state = RunState(path_map=tuple((str(a), str(b)) for a, b in path_map))
    patches = _Patches()
    previous = torch.empty(()).device
    STATE = state
    try:
        torch.set_default_device(META)
        with _DeviceMode(), _MetaDispatchMode():
            yield state
    finally:
        patches.restore()
        torch.set_default_device(previous)
        STATE = None


@dataclass(frozen=True)
class Outcome:
    """How a run ended. `rule` is None when it ran to completion."""

    rule: str | None
    reason: str
    state: str


def classify(exc: BaseException) -> Outcome:
    """Sort what ended a run into exactly one outcome."""
    if isinstance(exc, SystemExit):
        if exc.code in (None, 0):
            return Outcome(None, "", "completed")
        return Outcome("run-stopped", f"the script exited with status {exc.code}", "stopped")
    if isinstance(exc, MetaValueRead):
        return Outcome(
            "run-stopped",
            f"{exc.what} needs a tensor's value, which a meta tensor does not have",
            "stopped",
        )
    if isinstance(exc, Unsupported):
        return Outcome("run-unsupported", exc.what, "unsupported")
    if is_meta_value_error(exc):
        return Outcome(
            "run-stopped",
            "this needs a tensor's value, which a meta tensor does not have",
            "stopped",
        )
    blocked = unwrap_blocked(exc)
    if blocked is not None:
        if from_loader(exc):
            return Outcome(
                "run-stopped",
                f"needs files that are not available locally ({blocked.message})",
                "stopped",
            )
        # Review Focus 1: a script that writes its results at the end is not
        # buggy; the run just cannot go past the write.
        return Outcome(
            "run-stopped",
            f"writes and network are blocked during a run ({blocked.message})",
            "stopped",
        )
    if isinstance(exc, FileNotFoundError) or (isinstance(exc, OSError) and from_loader(exc)):
        return Outcome("run-stopped", f"missing input: {exc}", "stopped")
    text = str(exc)
    if any(message in text for message in _NO_CUDA_MESSAGES):
        return Outcome(
            "run-unsupported",
            f"this needs a real CUDA device: {text.splitlines()[0]}",
            "unsupported",
        )
    if isinstance(exc, NotImplementedError) and "meta" in text.lower():
        return Outcome("run-unsupported", text.splitlines()[0], "unsupported")
    return Outcome("run-error", "", "error")
