"""The meta runtime, in-process. Every test installs and uninstalls it."""

from __future__ import annotations

import builtins
import io

import pytest
import torch

from torchtyc import metaruntime
from torchtyc.effects import BlockedEffect
from torchtyc.metaruntime import MetaValueRead, Unsupported, classify


def _patch_targets():
    return (
        torch.load,
        torch.compile,
        torch.cuda.is_available,
        torch.cuda.memory_summary,
        builtins.open,
        io.open,
    )


@pytest.fixture
def runtime():
    before = _patch_targets()
    with metaruntime.install() as state:
        yield state
    # Every patch comes back out, by identity, even when the test failed.
    assert all(a is b for a, b in zip(_patch_targets(), before))
    assert metaruntime.STATE is None
    assert torch.empty(()).device.type == "cpu"


def test_factories_land_on_meta(runtime):
    assert torch.zeros(3).is_meta
    assert torch.zeros(3, device="cuda").is_meta
    assert torch.tensor([1, 2], device=torch.device("cuda", 0)).is_meta
    assert torch.empty(2).new_zeros(4, device="cuda").is_meta
    assert torch.nn.Linear(4, 5).weight.is_meta


def test_device_moves_stay_on_meta(runtime):
    x = torch.empty(2, 3)
    assert x.to("cuda:0").is_meta
    assert x.to(0).is_meta
    assert x.cuda().is_meta
    assert x.cpu().is_meta
    assert x.pin_memory().is_meta
    assert x.to("cuda", torch.float16).dtype is torch.float16
    assert torch.nn.Linear(2, 2).to("cuda", torch.bfloat16).weight.dtype is torch.bfloat16


def test_cpu_tensors_mix_with_meta(runtime):
    meta = torch.empty(4, 3)
    cpu = torch.ones(4, 3, device="cpu")
    assert (cpu + meta).is_meta
    assert torch.cat([torch.ones(1, 3, device="cpu"), meta]).shape == (5, 3)


VALUE_READS = {
    "item": lambda x: x.sum().item(),
    "bool": lambda x: bool(x.sum() > 0),
    "slice by tensor": lambda x: x[: torch.tensor(2)],
    "mask": lambda x: x[x > 0],
    "where one arg": lambda x: torch.where(x > 0),
    "nonzero": lambda x: x.nonzero(),
    "masked_select": lambda x: x.masked_select(x > 0),
    "unique": lambda x: torch.unique(x),
    "unique_consecutive": lambda x: torch.unique_consecutive(x),
    "bincount": lambda x: torch.bincount(x.long().flatten()),
    "equal": lambda x: torch.equal(x, x),
    "repeat_interleave by tensor": lambda x: torch.repeat_interleave(
        torch.ones(3, dtype=torch.long)
    ),
    "tolist": lambda x: x.tolist(),
    "numpy": lambda x: x.numpy(),
    "format": lambda x: f"{x.sum():.3f}",
}


@pytest.mark.parametrize("name", sorted(VALUE_READS))
def test_value_reads_stop(runtime, name):
    x = torch.empty(4, 3)
    with pytest.raises(MetaValueRead):
        VALUE_READS[name](x)


def test_value_read_escapes_except_exception(runtime):
    x = torch.empty(3)
    with pytest.raises(MetaValueRead):
        try:
            x.sum().item()
        except Exception:  # noqa: BLE001
            pytest.fail("a user's except Exception must not swallow the stop")


def test_shape_only_ops_do_not_stop(runtime):
    x = torch.empty(7, 5)
    assert x[torch.empty(3, dtype=torch.long)].shape == (3, 5)
    assert x.repeat_interleave(2, dim=0).shape == (14, 5)
    assert torch.repeat_interleave(
        x, torch.empty(7, dtype=torch.long), dim=0, output_size=9
    ).shape == (9, 5)
    assert f"{x}".startswith("tensor(")
    x[x > 0] = 0.0  # a masked assignment keeps the shape
    assert x.shape == (7, 5)


def test_allclose_is_recognised_by_message(runtime):
    x = torch.empty(3)
    with pytest.raises(BaseException) as info:
        torch.allclose(x, x)
    assert isinstance(info.value, MetaValueRead) or metaruntime.is_meta_value_error(info.value)


def test_install_twice_is_refused():
    with metaruntime.install(), pytest.raises(RuntimeError), metaruntime.install():
        pass


def test_remap():
    state = metaruntime.RunState(
        path_map=(("/workspace/golden", "/data/g"), ("/workspace", "/home/u/p"))
    )
    assert state.remap("/workspace/golden/a.st") == "/data/g/a.st"
    assert state.remap("/workspace/ckpt") == "/home/u/p/ckpt"
    assert state.remap("/workspacex/a") == "/workspacex/a"
    from pathlib import Path

    assert state.remap(Path("/workspace/x")) == Path("/home/u/p/x")
    assert state.remap(io.BytesIO()) is not None


def test_loading_marks_errors():
    state = metaruntime.RunState()
    with pytest.raises(OSError) as info, state.loading():
        raise OSError("no config.json")
    assert metaruntime.from_loader(info.value)


@pytest.mark.parametrize(
    ("exc", "rule", "state"),
    [
        (SystemExit(0), None, "completed"),
        (SystemExit(None), None, "completed"),
        (SystemExit(2), "run-stopped", "stopped"),
        (MetaValueRead("`.item()`"), "run-stopped", "stopped"),
        (Unsupported("no kernel"), "run-unsupported", "unsupported"),
        (RuntimeError("Tensor.item() cannot be called on meta tensors"), "run-stopped", "stopped"),
        (FileNotFoundError(2, "No such file", "/x/inputs.safetensors"), "run-stopped", "stopped"),
        (BlockedEffect("cannot write to 'notes/x.json' during run"), "run-stopped", "stopped"),
        (RuntimeError("Found no NVIDIA driver on your system."), "run-unsupported", "unsupported"),
        (
            NotImplementedError("Could not run 'aten::foo' with arguments from the 'Meta' backend"),
            "run-unsupported",
            "unsupported",
        ),
        (ValueError("rank mismatch"), "run-error", "error"),
    ],
)
def test_classify(exc, rule, state):
    outcome = classify(exc)
    assert (outcome.rule, outcome.state) == (rule, state)


def test_classify_oserror_from_loader_is_stopped():
    exc = OSError("does not appear to have a file named config.json")
    exc._torchtyc_loader = True
    assert classify(exc).rule == "run-stopped"
    assert classify(OSError("disk on fire")).rule == "run-error"
