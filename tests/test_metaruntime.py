"""The meta runtime, in-process. Every test installs and uninstalls it."""

from __future__ import annotations

import builtins
import importlib
import io
import sys
from pathlib import Path

import pytest
import torch
from hf_models import (
    TINY_VOCAB,
    requires_transformers,
    write_inputs,
    write_remote_checkpoint,
    write_tiny_llama,
    write_tokenizer,
)

from torchtyc import metaruntime
from torchtyc.effects import BlockedEffect
from torchtyc.metaruntime import MetaValueRead, Unsupported, classify


def _patch_targets():
    return (
        torch.load,
        torch.compile,
        torch.autocast,
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


def test_classify_value_error_from_loader_is_error():
    state = metaruntime.RunState()
    with pytest.raises(ValueError) as info, state.loading():
        raise ValueError("corrupt config")
    exc = info.value
    assert classify(exc).rule == "run-error"


def test_cuda_looks_present(runtime):
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.current_device() == 0
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory == 80 * (1 << 30)
    assert torch.cuda.mem_get_info()[0] == 80 * (1 << 30)
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(0)
    with torch.cuda.stream(torch.cuda.Stream()):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
    assert torch.cuda.current_stream().query()
    assert torch.cuda.is_current_stream_capturing() is False


def test_unstubbed_cuda_calls_are_unsupported(runtime):
    with pytest.raises(Unsupported):
        torch.cuda.memory_summary()


def test_unstubbed_cuda_probes_answer_no():
    import inspect

    probes = {
        name
        for name, value in vars(torch.cuda).items()
        if name.startswith("is_")
        and (inspect.isfunction(value) or inspect.isbuiltin(value))
        and str(getattr(value, "__module__", "")).startswith("torch.cuda")
    }
    stubbed = {"is_available", "is_bf16_supported", "is_initialized", "is_current_stream_capturing"}
    with metaruntime.install():
        for name in probes - stubbed:
            assert getattr(torch.cuda, name)() is False


def test_cuda_is_restored():
    real = torch.cuda.is_available
    with metaruntime.install():
        assert torch.cuda.is_available is not real
    assert torch.cuda.is_available is real


def test_torch_compile_runs_uncompiled(runtime):
    def f(x):
        return x * 2

    assert torch.compile(f) is f
    assert torch.compile(mode="max-autotune")(f) is f
    model = torch.nn.Linear(2, 2)
    assert torch.compile(model) is model
    assert any("torch.compile" in item for item in runtime.active)


def test_autocast_meta_is_noop(runtime):
    with torch.autocast(device_type="meta", enabled=False):
        y = torch.empty(2, 3) @ torch.empty(3, 4)
    assert y.shape == (2, 4)
    with torch.amp.autocast("meta"):
        pass
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pass


def test_autocast_is_restored():
    real = torch.autocast
    with metaruntime.install():
        assert torch.autocast is not real
    assert torch.autocast is real


def test_from_loader_walks_both_cause_and_context():
    state = metaruntime.RunState()
    with pytest.raises(RuntimeError) as info:
        try:
            with state.loading():
                raise OSError("missing weights")
        except OSError:
            raise RuntimeError("bad config") from KeyError("x")
    exc = info.value
    assert metaruntime.from_loader(exc)
    assert classify(exc).rule == "run-stopped"


def test_masked_index_put_on_real_tensor_skips_fast_path():
    before = _patch_targets()
    cpu = torch.ones(4)
    with metaruntime.install():
        assert not cpu.is_meta
        cpu[cpu > 0] = 0.0
        out = torch.index_put(cpu, (cpu > 0,), torch.zeros(4))
        assert out.is_meta
    assert all(a is b for a, b in zip(_patch_targets(), before))
    assert metaruntime.STATE is None
    assert torch.empty(()).device.type == "cpu"


def test_safetensors_reads_only_the_header(tmp_path):
    path = write_inputs(tmp_path / "inputs.safetensors", {"chat": 113, "long": 4096})
    with metaruntime.install():
        from safetensors.torch import load_file

        tensors = load_file(str(path))
    assert {k: tuple(v.shape) for k, v in tensors.items()} == {"chat": (113,), "long": (4096,)}
    assert all(v.is_meta and v.dtype is torch.int64 for v in tensors.values())


def test_safe_open_and_slices(tmp_path):
    path = write_inputs(tmp_path / "i.safetensors", {"a": 10})
    with metaruntime.install():
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            assert handle.keys() == ["a"]
            assert handle.get_tensor("a").shape == (10,)
            piece = handle.get_slice("a")
            assert piece.get_shape() == [10]
            assert piece.get_dtype() == "I64"
            assert piece[2:5].shape == (3,)


def test_path_map_applies_to_loaders_and_open(tmp_path):
    real = tmp_path / "golden"
    real.mkdir()
    write_inputs(real / "inputs.safetensors", {"a": 4})
    (real / "notes.txt").write_text("hello")
    with metaruntime.install([("/workspace/golden", str(real))]):
        from safetensors.torch import load_file

        assert load_file("/workspace/golden/inputs.safetensors")["a"].shape == (4,)
        with open("/workspace/golden/notes.txt") as handle:
            assert handle.read() == "hello"
        assert Path("/workspace/golden/notes.txt").read_text() == "hello"


def test_missing_input_is_marked_as_loader_error(tmp_path):
    with metaruntime.install():
        from safetensors.torch import load_file

        with pytest.raises(OSError) as info:
            load_file(str(tmp_path / "nope.safetensors"))
    assert metaruntime.classify(info.value).rule == "run-stopped"


def test_torch_load_path_and_bytes(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"w": torch.ones(3, 4)}, path)
    blob = io.BytesIO(path.read_bytes())
    with metaruntime.install():
        from_path = torch.load(path)
        from_bytes = torch.load(blob)
    assert from_path["w"].is_meta and from_path["w"].shape == (3, 4)
    assert from_bytes["w"].is_meta


def test_library_imported_before_install_is_patched(tmp_path):
    import safetensors.torch

    real = safetensors.torch.load_file
    with metaruntime.install():
        assert safetensors.torch.load_file is not real
    assert safetensors.torch.load_file is real


def test_library_imported_during_install_is_patched(tmp_path):
    # Only the Python modules: the Rust extension (safetensors._safetensors_rust)
    # stays cached, because a PyO3 module cannot be initialised twice.
    for name in ("safetensors.torch", "safetensors"):
        sys.modules.pop(name, None)
    with metaruntime.install():
        module = importlib.import_module("safetensors.torch")
        assert module.load_file.__module__ == "torchtyc.metaruntime"
    assert (
        importlib.import_module("safetensors.torch").load_file.__module__ != "torchtyc.metaruntime"
    )


def test_open_is_restored():
    real = builtins.open
    with metaruntime.install([("/a", "/b")]):
        assert builtins.open is not real
    assert builtins.open is real


@requires_transformers
def test_from_pretrained_builds_on_meta(tmp_path):
    path = write_tiny_llama(tmp_path / "m")
    with metaruntime.install() as state:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        assert all(p.is_meta for p in model.parameters())
        assert next(model.parameters()).dtype is torch.bfloat16
        assert not model.training
        assert {"device_map", "low_cpu_mem_usage"} <= state.ignored
        assert state.model_config["vocab_size"] == TINY_VOCAB


@requires_transformers
def test_concrete_class_from_pretrained(tmp_path):
    path = write_tiny_llama(tmp_path / "m")
    with metaruntime.install():
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        assert next(model.parameters()).is_meta
        assert next(model.parameters()).dtype is torch.float16


@requires_transformers
def test_auto_does_not_go_through_concrete_from_pretrained(tmp_path):
    path = write_tiny_llama(tmp_path / "m")
    with metaruntime.install():
        from transformers import AutoModelForCausalLM, PreTrainedModel

        calls = []
        patched = vars(PreTrainedModel)["from_pretrained"]
        # Set and restored by hand, inside the install block: a monkeypatch
        # would put the patched classmethod back after install() removed it.
        PreTrainedModel.from_pretrained = classmethod(
            lambda cls, *a, **k: calls.append(1) or patched.__func__(cls, *a, **k)
        )
        try:
            AutoModelForCausalLM.from_pretrained(path)
        finally:
            PreTrainedModel.from_pretrained = patched
        assert calls == []


@requires_transformers
def test_remote_code_with_cold_modules_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))
    path = write_remote_checkpoint(tmp_path / "remote")
    with metaruntime.install():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        out = model(input_ids=torch.zeros(1, 7, dtype=torch.long))
        assert type(out).__name__ == "CausalLMOutputWithPast"
        assert out.logits.shape == (1, 7, TINY_VOCAB)


@requires_transformers
def test_missing_checkpoint_is_stopped(tmp_path):
    with metaruntime.install():
        from transformers import AutoModelForCausalLM

        with pytest.raises(BaseException) as info:
            AutoModelForCausalLM.from_pretrained(tmp_path / "nowhere")
    assert metaruntime.classify(info.value).rule == "run-stopped"


@requires_transformers
def test_tokenizer_output_reaches_a_meta_model(tmp_path):
    model_path = write_tiny_llama(tmp_path / "m")
    tok_path = write_tokenizer(tmp_path / "t")
    with metaruntime.install():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
        batch = tokenizer(["w1 w2 w3", "w4"], return_tensors="pt", padding=True).to("cuda")
        assert batch.keys() == {"input_ids", "attention_mask"}
        logits = model(**batch, use_cache=False).logits
        assert logits.is_meta and logits.shape == (2, 3, TINY_VOCAB)


def _masked_forward(path, attn: str, hf_flags: bool) -> str:
    """Run a masked forward and say how it ended: "completed" or a rule."""
    with metaruntime.install(hf_flags=hf_flags):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path, attn_implementation=attn)
        ids = torch.zeros(1, 9, dtype=torch.long)
        mask = torch.ones(1, 9, dtype=torch.long)
        try:
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        except BaseException as exc:  # noqa: BLE001
            return metaruntime.classify(exc).rule or "completed"
        assert logits.shape == (1, 9, TINY_VOCAB)
        return "completed"


@requires_transformers
def test_mask_path_canary(tmp_path):
    """The two transformers flags must still matter.

    With them on, a masked forward completes on every attention
    implementation. With them off, at least one implementation stops on a
    value read inside HF's mask code (which one depends on the release: 4.45
    reads the mask only on the sdpa path). If an upgrade turns the flags into
    no-ops, the "off" runs stop being different and this fails.
    """
    path = write_tiny_llama(tmp_path / "m")
    on = {attn: _masked_forward(path, attn, hf_flags=True) for attn in ("eager", "sdpa")}
    off = {attn: _masked_forward(path, attn, hf_flags=False) for attn in ("eager", "sdpa")}
    assert on == {"eager": "completed", "sdpa": "completed"}
    assert "run-stopped" in off.values()
