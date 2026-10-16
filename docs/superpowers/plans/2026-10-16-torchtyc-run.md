# `torchtyc run` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `torchtyc run file.py[::func]`, which runs a user's real, unannotated entry point with every tensor on the meta device and reports shape bugs at the user's line.

**Architecture:** A new worker-only module, `metaruntime.py`, installs the meta runtime as one context manager. It sets the default device to meta, adds a TorchFunctionMode (devices, Python-level value reads) and a TorchDispatchMode (value-reading ops, CPU→meta moves, missing kernels), and patches the loaders, `from_pretrained`, the CUDA API and two transformers flags. The worker runs each entry in a forked child and streams diagnostics and heartbeats. The engine adds a run phase to `check_paths` with a hard per-entry deadline, suppressions in every file a diagnostic lands in, and a per-entry status.

**Tech Stack:** Python 3.11–3.13, torch ≥ 2.1 (dev: 2.13), transformers 4.45.2 / 4.57.6 / 5.20.0 (CI matrix), safetensors, tokenizers, einops, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-10-16-torchtyc-run-design.md`

## Global Constraints

- torch is imported only in `tracing.py` and `metaruntime.py`, and only the worker loads either. `cli.py`, `engine.py`, `config.py`, `formats.py`, `lsp.py` and `diagnostics.py` must never import torch, transformers or safetensors.
- Optional dependencies (transformers, safetensors, tokenizers, einops) are imported only inside patch callbacks or tests, never at module top level.
- Rule names are public API: `run-error` (error), `run-stopped` (info), `run-unsupported` (warning). They are added to `RULES` and to the README rules table.
- Config keys: `run` (list of `"file.py::func"` strings or `{ entry = "...", args = [...] }` tables), `path-map` (table of prefix → prefix), `run-timeout` (float seconds, default `300.0`). A bad value keeps its default and never raises (the existing `load()` contract).
- For `torchtyc run`, `--timeout` overrides `run-timeout`. The existing `timeout` key keeps its meaning for the trace pass.
- A value read never produces `run-error`; it is `run-stopped` (info). An unsupported op or CUDA call is `run-unsupported` (warning). Only an exception from the user's own logic is `run-error`.
- `MetaValueRead` and `Unsupported` subclass `BaseException`.
- No test touches the network or downloads a model. Every test that touches transformers runs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, a temporary `HF_HOME` outside the test's project tree, and socket `connect` disabled.
- Diagnostics are 0-based `line`/`column`; formats add 1 when printing.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Use `uv add` for dependencies, never `uv pip install`. Run tests with `just test ...` or `uv run pytest ...`.
- **Refinement of the spec (§2, `check` and config entries):** `check` runs a config `run` entry only when the entry's file is inside one of the paths being checked. `torchtyc check src/` then doesn't run an entry that lives in `tests/fixtures/`, while `torchtyc check tests/fixtures` does. Without this, the repo's own self-check entry would make every `torchtyc check src` exit 1.
- **Refinement of the spec (§4.2):** the Python-level value-read checks (`tolist`, `numpy`, `__array__`, `__format__`, `item`) live in the TorchFunctionMode, not in method patches. A probe showed every one of them passes through `__torch_function__`, so nothing on `torch.Tensor` needs patching or restoring.
- **Refinement of the spec (§6 canary):** the HF mask-path canary is an in-process test in `tests/test_metaruntime.py`, using `install(hf_flags=False)`. A worker job flag would exist only for this test.

## Review Focus

Five inputs the spec implies but no section tests, most likely first. Each one has a test in the task that owns the code.

1. **The script writes its results at the end** (`json.dump(..., open("notes/x.json", "w"))`, `torch.save`), as the motivating script's docstring plans to. The effects guard blocks the write. A reasonable person expects that to end the run with info-level `run-stopped` ("writes are blocked during a run"), not a red `run-error`. **This changes the spec's §3.3 row "`BlockedEffect` anywhere else → the existing diagnostic"** to `run-stopped` with the BlockedEffect message. The user has to confirm it at the plan review (see the handoff). Tested in Task 4 (classify) and Task 8 (end-to-end).
2. **`async def main()`.** Calling it returns a coroutine, and the run would "complete" with nothing executed. Expected: the coroutine is awaited with `asyncio.run`. Tested in Task 8.
3. **The entry file has a syntax error.** Expected: a `run-error` at the line of the syntax error, not line 1. Tested in Task 8.
4. **The script prints a lot to stdout.** Expected: the JSON channel stays intact and the run is reported normally. Tested in Task 8.
5. **`torchtyc run tools/x.py::main` from a working directory other than the project root.** Expected: the entry resolves against the current directory and the report uses relative paths. Tested in Task 11.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/torchtyc/metaruntime.py` (new) | The meta runtime: exceptions, `RunState`, the two modes, the patch registry, CUDA/compile stubs, the import hook, loaders, transformers and einops patches, labels, `classify()`, `install()` |
| `src/torchtyc/worker.py` | `mode: "run"` jobs: a child per entry, streamed events, heartbeats, entry loading (`::func`, bare file, package), the run diagnostic (anchoring, message, `got`, `hint`, `via`) |
| `src/torchtyc/engine.py` | `RunStatus`, `EntryError`, `entry_problem()`, `_run_worker_entries()`, the run phase of `check_paths`, suppressions in extra files, dedupe |
| `src/torchtyc/config.py` | `RunEntry`, `run`, `path-map`, `run-timeout`, `Overrides.run_timeout` |
| `src/torchtyc/diagnostics.py` | three rules; `Diagnostic.via` |
| `src/torchtyc/effects.py` | `shutil` guards check destinations; HF cache directories |
| `src/torchtyc/formats.py` | status lines, `run_status` in json, `via` |
| `src/torchtyc/cli.py` | the `run` command, `--` passthrough, config entries for `check` |
| `tests/conftest.py` (new) | autouse hermetic HF environment |
| `tests/hf_models.py` (new) | writers for tiny Llama, remote-code checkpoint, tokenizer, safetensors inputs |
| `tests/test_metaruntime.py` (new) | unit tests, in-process |
| `tests/test_run.py` (new) | end-to-end through the real worker |
| `tests/fixtures/run_bugs.py` (new) | self-check fixture |

---

### Task 1: Dev dependencies, hermetic HF test helpers, transformers compatibility probe

**Files:**
- Modify: `pyproject.toml` (dev group, via `uv add`), `uv.lock`
- Create: `tests/conftest.py`, `tests/hf_models.py`
- Scratch (not committed): `$SCRATCH/probe_hf.py`

**Interfaces:**
- Produces: the autouse fixture `hf_environment` (no direct use needed). From `tests/hf_models.py`: `requires_transformers` (a pytest mark); `write_tiny_llama(dest: Path, **overrides) -> Path`; `write_remote_checkpoint(dest: Path) -> Path`; `write_tokenizer(dest: Path) -> Path`; `write_inputs(path: Path, lengths: dict[str, int]) -> Path`; the constants `TINY_VOCAB = 1000` and `TINY_HIDDEN = 64`.

- [ ] **Step 1: Add the dev dependencies**

```bash
uv add --dev "transformers>=4.45" "safetensors>=0.4" "tokenizers>=0.20"
```

Expected: `pyproject.toml`'s `[dependency-groups] dev` gains the three lines, and `uv.lock` updates. transformers pulls in numpy.

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Keep every test off the network and out of the user's Hugging Face cache.

The HF home lives in its own temporary directory, not under the test's
`tmp_path`: end-to-end tests make `tmp_path` the project root, and the effects
guard blocks writes inside the project tree, which would block the remote-code
modules cache too.
"""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def hf_environment(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("hf_home")
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    for name in ("HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(name, raising=False)

    def refuse(self, address):
        raise OSError(f"network disabled in tests (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    return home
```

- [ ] **Step 3: Write `tests/hf_models.py`**

```python
"""Tiny Hugging Face checkpoints written from scratch, so no test downloads one.

Everything here goes through public, long-stable transformers API
(`LlamaConfig`, `PretrainedConfig`, `PreTrainedModel`, `PreTrainedTokenizerFast`),
so the same files load on the 4.x and 5.x releases the CI matrix pins.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import textwrap
from pathlib import Path

import pytest

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="needs transformers"
)

TINY_VOCAB = 1000
TINY_HIDDEN = 64


def write_tiny_llama(dest: Path, **overrides) -> Path:
    from transformers import LlamaConfig

    values = dict(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    values.update(overrides)
    dest.mkdir(parents=True, exist_ok=True)
    LlamaConfig(**values).save_pretrained(dest)
    return dest


_REMOTE_CONFIG = '''
from transformers import PretrainedConfig


class TinyRemoteConfig(PretrainedConfig):
    model_type = "tiny_remote"

    def __init__(self, vocab_size=1000, hidden_size=64, num_hidden_layers=2, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        super().__init__(**kwargs)
'''

_REMOTE_MODEL = '''
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_tiny_remote import TinyRemoteConfig


class TinyRemoteForCausalLM(PreTrainedModel):
    config_class = TinyRemoteConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_hidden_layers)
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, use_cache=None, **kwargs):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = torch.relu(layer(hidden))
        return CausalLMOutputWithPast(logits=self.lm_head(hidden))
'''


def write_remote_checkpoint(dest: Path) -> Path:
    """A checkpoint that needs `trust_remote_code`, like the motivating K2 model."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "configuration_tiny_remote.py").write_text(_REMOTE_CONFIG)
    (dest / "modeling_tiny_remote.py").write_text(_REMOTE_MODEL)
    (dest / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tiny_remote",
                "architectures": ["TinyRemoteForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_tiny_remote.TinyRemoteConfig",
                    "AutoModelForCausalLM": "modeling_tiny_remote.TinyRemoteForCausalLM",
                },
                "vocab_size": TINY_VOCAB,
                "hidden_size": TINY_HIDDEN,
                "num_hidden_layers": 2,
            }
        )
    )
    return dest


def write_tokenizer(dest: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, **{f"w{i}": i + 2 for i in range(50)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    dest.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(dest)
    return dest


def write_inputs(path: Path, lengths: dict[str, int]) -> Path:
    """A safetensors file of int64 token ids, written by hand so it needs no torch."""
    header: dict[str, dict] = {}
    offset = 0
    for name, length in lengths.items():
        size = length * 8
        header[name] = {"dtype": "I64", "shape": [length], "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    return path


def dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")
```

- [ ] **Step 4: Check the helpers work and the suite still passes**

Run: `uv run python -c "import sys; sys.path.insert(0, 'tests'); import hf_models, tempfile, pathlib; d = pathlib.Path(tempfile.mkdtemp()); hf_models.write_tiny_llama(d / 'm'); hf_models.write_tokenizer(d / 't'); hf_models.write_remote_checkpoint(d / 'r'); hf_models.write_inputs(d / 'i.safetensors', {'a': 5}); print(sorted(p.name for p in d.rglob('*')))"`
Expected: a list including `config.json`, `tokenizer.json`, `modeling_tiny_remote.py` and `i.safetensors`.

Run: `just test -q`
Expected: every existing test passes (the autouse fixture must not break them).

- [ ] **Step 5: Probe `is_torchdynamo_compiling → True` against the three pinned releases (spec §7, risk 1)**

Write `$SCRATCH/probe_hf.py`, where `$SCRATCH` is your scratch directory, not the repo:

```python
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "tests")
import hf_models  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

real_tracing = getattr(transformers.utils.import_utils, "is_tracing", None)
real_compiling = transformers.utils.import_utils.is_torchdynamo_compiling
for name, module in list(sys.modules.items()):
    if name.startswith("transformers"):
        for attr, value in list(vars(module).items()):
            if value is real_compiling:
                setattr(module, attr, lambda: True)
            if real_tracing is not None and value is real_tracing:
                setattr(module, attr, lambda t=None: (isinstance(t, torch.Tensor) and t.is_meta) or real_tracing(t))

path = hf_models.write_tiny_llama(Path(tempfile.mkdtemp()) / "m")
config = AutoConfig.from_pretrained(path)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
ids = torch.zeros(1, 17, dtype=torch.long, device="meta")
mask = torch.ones(1, 17, dtype=torch.long, device="meta")
out = model(input_ids=ids, attention_mask=mask, use_cache=False)
print(transformers.__version__, "OK", tuple(out.logits.shape))
```

Run each:
```bash
for v in 4.45.2 4.57.6 5.20.0; do HF_HUB_OFFLINE=1 uv run --with "transformers==$v" python "$SCRATCH/probe_hf.py" || echo "FAIL $v"; done
```
Expected: three lines `<version> OK (1, 17, 1000)`.

If a version fails, decide by the error:
- **The failure is in the probe's setup** (that transformers doesn't import with torch 2.13): raise the oldest pin to the first 4.x minor that imports, and record the new pin in this plan's Tech Stack line and in Task 13's CI matrix.
- **The forward fails because of the `is_torchdynamo_compiling` patch** (it reaches a dynamo-only API): follow spec §7, risk 1. Drop the `is_torchdynamo_compiling` patch from Task 7; Task 7's canary then expects 4.x to stop inside HF, with `transformers is_torchdynamo_compiling` reported as inactive.

Write the outcome in the Task 1 commit message.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py tests/hf_models.py
git commit -m "test: hermetic Hugging Face fixtures for the run command

Probe: is_torchdynamo_compiling -> True works on <record results here>.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Rules, config keys, `Diagnostic.via`

**Files:**
- Modify: `src/torchtyc/diagnostics.py`, `src/torchtyc/config.py`
- Test: `tests/test_config.py`, `tests/test_formats.py` (one round-trip assertion)

**Interfaces:**
- Produces: `RULES["run-error"|"run-stopped"|"run-unsupported"]`; `Diagnostic.via: str | None`; `config.RunEntry(path: str, func: str | None = None, args: tuple[str, ...] = ())` with `.spec -> str` and `RunEntry.parse(text: str, args: Sequence[str] = ()) -> RunEntry`; `Config.run: tuple[RunEntry, ...]`, `Config.path_map: tuple[tuple[str, str], ...]` (longest prefix first, targets absolute), `Config.run_timeout: float = 300.0`; `Overrides.run_timeout: float | None`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_config.py`)

```python
def test_run_keys_are_read(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        run = ["tools/a.py::main", { entry = "tools/b.py", args = ["--n", "2"] }]
        path-map = { "/workspace" = ".", "/workspace/golden" = "data/golden" }
        run-timeout = 12
        """,
    )
    config = config_module.load(root)
    assert [e.spec for e in config.run] == [
        str(root / "tools/a.py") + "::main",
        str(root / "tools/b.py"),
    ]
    assert config.run[1].args == ("--n", "2")
    # Longest prefix first, so the more specific mapping wins.
    assert config.path_map == (
        ("/workspace/golden", str((root / "data/golden").resolve())),
        ("/workspace", str(root.resolve())),
    )
    assert config.run_timeout == 12.0


def test_bad_run_keys_keep_defaults(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        run = "not-a-list"
        path-map = ["nope"]
        run-timeout = -1
        """,
    )
    config = config_module.load(root)
    assert config.run == ()
    assert config.path_map == ()
    assert config.run_timeout == 300.0


def test_run_entry_parse():
    entry = config_module.RunEntry.parse("x.py::main", ["--a"])
    assert (entry.path, entry.func, entry.args, entry.spec) == ("x.py", "main", ("--a",), "x.py::main")
    bare = config_module.RunEntry.parse("x.py")
    assert (bare.func, bare.spec) == (None, "x.py")


def test_run_timeout_override():
    config = config_module.Config(root=__import__("pathlib").Path("."))
    config_module.Overrides(run_timeout=3.0).apply(config)
    assert config.run_timeout == 3.0
    assert config.timeout == 60.0
```

Append to `tests/test_formats.py`:

```python
def test_diagnostic_via_round_trips():
    from torchtyc.diagnostics import RULES, Diagnostic

    d = Diagnostic(path="a.py", line=1, column=0, rule="run-error", message="m", via="main (line 2)")
    assert Diagnostic.from_json(d.to_json()).via == "main (line 2)"
    assert {"run-error", "run-stopped", "run-unsupported"} <= set(RULES)
```

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_config.py tests/test_formats.py -q`
Expected: FAIL. `RunEntry`, `run_timeout` and `via` don't exist yet.

- [ ] **Step 3: Implement the rules and `via`** (in `src/torchtyc/diagnostics.py`)

After `TRACE_RETRIED`, add:

```python
# `torchtyc run`: a script's real entry point, executed on meta tensors.
RUN_ERROR = _rule("run-error", Severity.ERROR, "the entry point raised while running on meta tensors")
RUN_STOPPED = _rule(
    "run-stopped", Severity.INFO, "the run could not continue for a reason that is not a bug"
)
RUN_UNSUPPORTED = _rule(
    "run-unsupported", Severity.WARNING, "the run reached something the meta runtime cannot stand in for"
)
```

In `Diagnostic`, after `suggestion`, add:

```python
    # The chain of project frames that led to a run failure, outermost first:
    # `main (line 47) → run (line 36)`.
    via: str | None = None
```

- [ ] **Step 4: Implement the config** (in `src/torchtyc/config.py`)

Add after the imports:

```python
@dataclass(frozen=True)
class RunEntry:
    """One entry point for `torchtyc run`: a file, an optional function, its argv."""

    path: str
    func: str | None = None
    args: tuple[str, ...] = ()

    @property
    def spec(self) -> str:
        return f"{self.path}::{self.func}" if self.func else self.path

    @classmethod
    def parse(cls, text: str, args: Sequence[str] = ()) -> RunEntry:
        path, _, func = text.partition("::")
        return cls(path=path, func=func or None, args=tuple(args))
```

In `Config`, after `allow_effects`, add:

```python
    # `torchtyc run`: entries `check` also runs, prefix rewrites for
    # hard-coded paths, and the wall-clock limit for one entry.
    run: tuple[RunEntry, ...] = ()
    path_map: tuple[tuple[str, str], ...] = ()
    run_timeout: float = 300.0
```

In `Overrides`, add the field `run_timeout: float | None = None`, and in `apply()`:

```python
        if self.run_timeout is not None:
            config.run_timeout = self.run_timeout
```

Add the readers above `load()`:

```python
def _run_entries(value: object, root: Path) -> tuple[RunEntry, ...]:
    if not isinstance(value, list):
        raise TypeError("run must be a list")
    entries: list[RunEntry] = []
    for item in value:
        if isinstance(item, str):
            text, args = item, ()
        elif isinstance(item, dict) and isinstance(item.get("entry"), str):
            text, args = item["entry"], tuple(_strings(item.get("args", [])))
        else:
            raise TypeError("each run entry is a string or a table with `entry`")
        entry = RunEntry.parse(text, args)
        path = Path(entry.path)
        entries.append(RunEntry(str(path if path.is_absolute() else root / path), entry.func, entry.args))
    return tuple(entries)


def _path_map(value: object, root: Path) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise TypeError("path-map must be a table")
    pairs = []
    for old, new in value.items():
        target = Path(str(new)).expanduser()
        pairs.append((str(old).rstrip("/") or "/", str((target if target.is_absolute() else root / target).resolve())))
    return tuple(sorted(pairs, key=lambda pair: -len(pair[0])))
```

At the end of `load()`, before `return config`:

```python
    read("run", "run", lambda v: _run_entries(v, root))
    read("path-map", "path_map", lambda v: _path_map(v, root))
    read("run-timeout", "run_timeout", lambda v: _positive(float(v)))
```

- [ ] **Step 5: Run the tests**

Run: `just test tests/test_config.py tests/test_formats.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/torchtyc/diagnostics.py src/torchtyc/config.py tests/test_config.py tests/test_formats.py
git commit -m "feat: rules and config keys for torchtyc run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Effects guard: `shutil` destinations and HF cache directories

**Files:**
- Modify: `src/torchtyc/effects.py` (`_scratch_dirs`, `_guarded_shutil_copy`, `_guarded_shutil_copy2`, `_guarded_shutil_copyfile`, `_guarded_shutil_copytree`, `_guarded_shutil_move`)
- Test: `tests/test_effects.py`

**Interfaces:**
- Produces: `shutil.copy*`/`copytree`/`move` succeed under the guard when the destination passes `_is_allowed_write`; `_scratch_dirs()` includes `HF_HOME`, `HF_HUB_CACHE`, `HF_MODULES_CACHE` and `TRANSFORMERS_CACHE` when set.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_effects.py`)

```python
import shutil

from torchtyc import effects


def test_shutil_copy_to_scratch_is_allowed(tmp_path: Path):
    src = tmp_path / "a.py"
    src.write_text("x = 1\n")
    dest = tmp_path / "cache" / "a.py"
    dest.parent.mkdir()
    with active_guard(enabled=True):
        shutil.copyfile(src, dest)
        shutil.copy(src, tmp_path / "cache" / "b.py")
        shutil.copy2(src, tmp_path / "cache" / "c.py")
    assert dest.read_text() == "x = 1\n"


def test_shutil_copy_into_protected_tree_is_blocked(tmp_path: Path):
    src = tmp_path / "a.py"
    src.write_text("x = 1\n")
    project = tmp_path / "project"
    project.mkdir()
    with active_guard(enabled=True, protect=[project]):
        with pytest.raises(BlockedEffect):
            shutil.copyfile(src, project / "a.py")
        with pytest.raises(BlockedEffect):
            shutil.copytree(tmp_path / "project", project / "nested")


def test_hf_cache_variables_are_scratch(monkeypatch, tmp_path: Path):
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
        target = tmp_path / name.lower()
        target.mkdir()
        monkeypatch.setenv(name, str(target))
    dirs = effects._scratch_dirs()
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
        assert (tmp_path / name.lower()).resolve() in dirs
```

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_effects.py -q -k "shutil or hf_cache"`
Expected: `test_shutil_copy_to_scratch_is_allowed` and `test_hf_cache_variables_are_scratch` FAIL; the protected-tree test passes already.

- [ ] **Step 3: Implement**

Replace the four copy guards and the move guard with destination checks (keep the existing messages for the blocked case):

```python
def _guarded_shutil_copy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
    if _is_allowed_write(dst):
        return _orig_shutil_copy(src, dst, *args, **kwargs)
    raise BlockedEffect(
        f"cannot copy to '{dst}' during {_current_phase()}", target=str(dst), culprit=_caller_info()
    )
```

Do the same for `copy2` (`_orig_shutil_copy2`), `copyfile` (`_orig_shutil_copyfile`), `copytree` (`_orig_shutil_copytree`) and `move` (`_orig_shutil_move`). Read the current body of `_guarded_shutil_move` first and keep its message text.

In `_scratch_dirs()`, before `return dirs`:

```python
    # Hugging Face caches: `trust_remote_code` copies modeling files into the
    # modules cache, and the hub writes lock files even on a cache hit.
    for env_var in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
        val = os.environ.get(env_var)
        if val:
            try:
                dirs.append(Path(val).expanduser().resolve())
            except (RuntimeError, ValueError, OSError):
                pass
```

Update the `_scratch_dirs` docstring to mention the HF variables.

- [ ] **Step 4: Run the tests**

Run: `just test tests/test_effects.py -q`
Expected: PASS, including every existing effects test.

- [ ] **Step 5: Commit**

```bash
git add src/torchtyc/effects.py tests/test_effects.py
git commit -m "fix: let the effects guard copy into allowed directories

The shutil guards blocked every copy, so trust_remote_code could never
populate the Hugging Face modules cache. They now check the destination
the same way open() does.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Meta runtime core: exceptions, modes, `install()`, `classify()`

**Files:**
- Create: `src/torchtyc/metaruntime.py`
- Test: `tests/test_metaruntime.py`

**Interfaces:**
- Produces:
  - `MetaValueRead(BaseException)` and `Unsupported(BaseException)`, each with `.what: str`.
  - `RunState`, with `remap(path) -> path`, `loading()` (a context manager), `active: list[str]`, `inactive: list[str]`, `ignored: set[str]`, `model_config: dict[str, int]`, `einops_axes: dict[str, set[int]]`.
  - `install(path_map=(), *, hf_flags: bool = True) -> ContextManager[RunState]`.
  - `Outcome(rule: str | None, reason: str, state: str)` and `classify(exc: BaseException) -> Outcome`, where `state` is one of `completed | stopped | unsupported | error`.
  - `from_loader(exc) -> bool`, `is_meta_value_error(exc) -> bool`, and the module-level `STATE: RunState | None`.
  - `_Patches` and `_state() -> RunState`, which later tasks use.

- [ ] **Step 1: Write the failing tests** (`tests/test_metaruntime.py`)

```python
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
    return (torch.load, torch.compile, torch.cuda.is_available, torch.cuda.memory_summary, builtins.open, io.open)


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
    "repeat_interleave by tensor": lambda x: torch.repeat_interleave(torch.ones(3, dtype=torch.long)),
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
    assert torch.repeat_interleave(x, torch.empty(7, dtype=torch.long), dim=0, output_size=9).shape == (9, 5)
    assert f"{x}".startswith("tensor(")
    x[x > 0] = 0.0  # a masked assignment keeps the shape
    assert x.shape == (7, 5)


def test_allclose_is_recognised_by_message(runtime):
    x = torch.empty(3)
    with pytest.raises(BaseException) as info:
        torch.allclose(x, x)
    assert isinstance(info.value, MetaValueRead) or metaruntime.is_meta_value_error(info.value)


def test_install_twice_is_refused():
    with metaruntime.install():
        with pytest.raises(RuntimeError):
            with metaruntime.install():
                pass


def test_remap():
    state = metaruntime.RunState(path_map=(("/workspace/golden", "/data/g"), ("/workspace", "/home/u/p")))
    assert state.remap("/workspace/golden/a.st") == "/data/g/a.st"
    assert state.remap("/workspace/ckpt") == "/home/u/p/ckpt"
    assert state.remap("/workspacex/a") == "/workspacex/a"
    from pathlib import Path

    assert state.remap(Path("/workspace/x")) == Path("/home/u/p/x")
    assert state.remap(io.BytesIO()) is not None


def test_loading_marks_errors():
    state = metaruntime.RunState()
    with pytest.raises(OSError) as info:
        with state.loading():
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
        (NotImplementedError("Could not run 'aten::foo' with arguments from the 'Meta' backend"), "run-unsupported", "unsupported"),
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
```

The `BlockedEffect` row encodes Review Focus 1.

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_metaruntime.py -q`
Expected: FAIL at collection: `No module named 'torchtyc.metaruntime'`.

- [ ] **Step 3: Implement `src/torchtyc/metaruntime.py`**

```python
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
            "run-stopped", f"{exc.what} needs a tensor's value, which a meta tensor does not have", "stopped"
        )
    if isinstance(exc, Unsupported):
        return Outcome("run-unsupported", exc.what, "unsupported")
    if is_meta_value_error(exc):
        return Outcome(
            "run-stopped", "this needs a tensor's value, which a meta tensor does not have", "stopped"
        )
    blocked = unwrap_blocked(exc)
    if blocked is not None:
        if from_loader(exc):
            return Outcome(
                "run-stopped", f"needs files that are not available locally ({blocked.message})", "stopped"
            )
        # Review Focus 1: a script that writes its results at the end is not
        # buggy; the run just cannot go past the write.
        return Outcome("run-stopped", f"writes and network are blocked during a run ({blocked.message})", "stopped")
    if isinstance(exc, FileNotFoundError) or (isinstance(exc, OSError) and from_loader(exc)):
        return Outcome("run-stopped", f"missing input: {exc}", "stopped")
    text = str(exc)
    if any(message in text for message in _NO_CUDA_MESSAGES):
        return Outcome("run-unsupported", f"this needs a real CUDA device: {text.splitlines()[0]}", "unsupported")
    if isinstance(exc, NotImplementedError) and "meta" in text.lower():
        return Outcome("run-unsupported", text.splitlines()[0], "unsupported")
    return Outcome("run-error", "", "error")
```

`patches` is unused until Task 5; keep it, because Task 5 adds the first call.

- [ ] **Step 4: Run the tests**

Run: `just test tests/test_metaruntime.py -q`
Expected: PASS. If `test_value_reads_stop[tolist]` or `[numpy]` fails, the function mode isn't seeing that method on this torch. Print `func` inside `_DeviceMode.__torch_function__` for that call and add the object you see to `_VALUE_METHODS`.

- [ ] **Step 5: Lint and commit**

```bash
just lint
git add src/torchtyc/metaruntime.py tests/test_metaruntime.py
git commit -m "feat: meta runtime core for torchtyc run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: CUDA stubs and `torch.compile`

**Files:**
- Modify: `src/torchtyc/metaruntime.py`
- Test: `tests/test_metaruntime.py`

**Interfaces:**
- Consumes: `_Patches`, `RunState`, `Unsupported`, `install()` from Task 4.
- Produces: `_patch_cuda(patches)` and `_patch_compile(patches, state)`, called from `install()`.

- [ ] **Step 1: Write the failing tests** (append)

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_metaruntime.py -q -k "cuda or compile"`
Expected: FAIL. `torch.cuda.is_available()` is False on a CPU host.

- [ ] **Step 3: Implement** (add to `metaruntime.py`, above `install`)

```python
import inspect

_GIB = 1 << 30


@dataclass(frozen=True)
class _FakeDeviceProperties:
    name: str = "torchtyc meta device"
    major: int = 9
    minor: int = 0
    total_memory: int = 80 * _GIB
    multi_processor_count: int = 132
    is_integrated: bool = False
    is_multi_gpu_board: bool = False
    warp_size: int = 32


_PROPS = _FakeDeviceProperties()


class _FakeEvent:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record(self, stream: Any = None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream: Any = None) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, other: Any) -> float:
        return 0.0


class _FakeStream:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def synchronize(self) -> None:
        pass

    def wait_stream(self, stream: Any) -> None:
        pass

    def wait_event(self, event: Any) -> None:
        pass

    def record_event(self, event: Any = None) -> Any:
        return event or _FakeEvent()

    def query(self) -> bool:
        return True


def _nothing(*args: Any, **kwargs: Any) -> None:
    return None


def _cuda_stubs() -> dict[str, Any]:
    stubs: dict[str, Any] = {
        "is_available": lambda: True,
        "is_initialized": lambda: True,
        "is_bf16_supported": lambda *a, **k: True,
        "is_current_stream_capturing": lambda: False,
        "device_count": lambda: 1,
        "current_device": lambda: 0,
        "get_device_name": lambda device=None: _PROPS.name,
        "get_device_capability": lambda device=None: (_PROPS.major, _PROPS.minor),
        "get_device_properties": lambda device=None: _PROPS,
        "mem_get_info": lambda device=None: (_PROPS.total_memory, _PROPS.total_memory),
        "memory_allocated": lambda device=None: 0,
        "max_memory_allocated": lambda device=None: 0,
        "memory_reserved": lambda device=None: 0,
        "max_memory_reserved": lambda device=None: 0,
        "current_stream": lambda device=None: _FakeStream(),
        "default_stream": lambda device=None: _FakeStream(),
        "stream": lambda stream=None: contextlib.nullcontext(stream),
        "device": lambda device=None: contextlib.nullcontext(),
        "get_rng_state": lambda device="cuda": torch.zeros(16, dtype=torch.uint8),
        "Stream": _FakeStream,
        "Event": _FakeEvent,
    }
    for name in (
        "init",
        "set_device",
        "synchronize",
        "empty_cache",
        "ipc_collect",
        "reset_peak_memory_stats",
        "reset_max_memory_allocated",
        "set_per_process_memory_fraction",
        "manual_seed",
        "manual_seed_all",
        "seed",
        "seed_all",
        "set_rng_state",
        "set_rng_state_all",
    ):
        stubs[name] = _nothing
    return stubs


def _unstubbed_cuda(name: str) -> Any:
    if name.startswith(("is_", "has_")):
        # A capability probe that answers "no" sends code down its portable path.
        def probe(*args: Any, **kwargs: Any) -> bool:
            return False

        return probe

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise Unsupported(f"`torch.cuda.{name}` needs a real CUDA device")

    return refuse


def _patch_cuda(patches: _Patches) -> None:
    """Make CUDA look present, the same way on every machine."""
    stubs = _cuda_stubs()
    for name, value in list(vars(torch.cuda).items()):
        if name.startswith("_") or name in stubs:
            continue
        if (inspect.isfunction(value) or inspect.isbuiltin(value)) and str(
            getattr(value, "__module__", "")
        ).startswith("torch.cuda"):
            patches.set(torch.cuda, name, _unstubbed_cuda(name))
    for name, value in stubs.items():
        patches.set(torch.cuda, name, value)


def _no_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
    if model is None:
        return lambda fn: fn
    return model


def _patch_compile(patches: _Patches, state: RunState) -> None:
    patches.set(torch, "compile", _no_compile)
    if "compile" in vars(torch.nn.Module):
        patches.set(torch.nn.Module, "compile", lambda self, *args, **kwargs: None)
    state.active.append("torch.compile (runs uncompiled)")
```

Move `import inspect` to the top-level imports. In `install()`, right after `torch.set_default_device(META)`:

```python
        _patch_cuda(patches)
        _patch_compile(patches, state)
```

- [ ] **Step 4: Run the tests**

Run: `just test tests/test_metaruntime.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
just lint
git add src/torchtyc/metaruntime.py tests/test_metaruntime.py
git commit -m "feat: CUDA stubs and an uncompiled torch.compile for runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Loaders: the import hook, safetensors, `torch.load`, `path-map` on `open`

**Files:**
- Modify: `src/torchtyc/metaruntime.py`
- Test: `tests/test_metaruntime.py`

**Interfaces:**
- Consumes: `_Patches`, `RunState.remap`, `RunState.loading`, `_state()`.
- Produces: `_ImportHook(patches)` with `.on_import(name, callback)`, `.install()` and `.remove()`; the patched `safetensors.safe_open`, `safetensors.torch.load_file/load/safe_open`, `torch.load` and `builtins.open`/`io.open`. Task 7 registers more callbacks on the same hook through `_register_library_patches(hook, patches, state, hf_flags)`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import importlib
import sys
from pathlib import Path

from hf_models import write_inputs


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
    assert importlib.import_module("safetensors.torch").load_file.__module__ != "torchtyc.metaruntime"


def test_open_is_restored():
    real = builtins.open
    with metaruntime.install([("/a", "/b")]):
        assert builtins.open is not real
    assert builtins.open is real
```

`tests/` is on `sys.path` under pytest's default rootdir conftest handling, which is how `from hf_models import ...` resolves. If it isn't, add `pythonpath = ["tests"]` to `[tool.pytest.ini_options]` in `pyproject.toml` (create the table if it's missing).

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_metaruntime.py -q -k "safetensors or safe_open or path_map or torch_load or library or open_is or missing_input"`
Expected: FAIL. The tensors are real CPU tensors, and the path-mapped open raises `FileNotFoundError`.

- [ ] **Step 3: Implement** (add to `metaruntime.py`)

New top-level imports: `builtins`, `importlib.abc`, `importlib.util`, `io`, `json`, `struct`, `sys`, `from collections.abc import Callable`, `from types import ModuleType`.

```python
# Captured before any patch, so header reads never go through path-map twice.
_real_open = io.open

_ST_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
for _code, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _attr):
        _ST_DTYPES[_code] = getattr(torch, _attr)


def _parse_header(raw: bytes) -> tuple[dict[str, dict], dict[str, str]]:
    header = json.loads(raw)
    metadata = header.pop("__metadata__", None) or {}
    return header, metadata


def _read_header(path: Any) -> tuple[dict[str, dict], dict[str, str]]:
    with _real_open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return _parse_header(handle.read(length))


def _meta_tensor(info: dict) -> torch.Tensor:
    return torch.empty(info["shape"], dtype=_ST_DTYPES[info["dtype"]], device=META)


class _MetaSlice:
    def __init__(self, info: dict) -> None:
        self._info = info

    def get_shape(self) -> list[int]:
        return list(self._info["shape"])

    def get_dtype(self) -> str:
        return self._info["dtype"]

    def __getitem__(self, index: Any) -> torch.Tensor:
        return _meta_tensor(self._info)[index]


class _MetaSafeOpen:
    """`safetensors.safe_open`, answered from the header alone."""

    def __init__(self, filename: Any, framework: str = "pt", device: Any = "cpu") -> None:
        state = _state()
        with state.loading():
            self._header, self._metadata = _read_header(state.remap(filename))

    def __enter__(self) -> _MetaSafeOpen:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def keys(self) -> list[str]:
        return list(self._header)

    def offset_keys(self) -> list[str]:
        return self.keys()

    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    def get_tensor(self, name: str) -> torch.Tensor:
        return _meta_tensor(self._header[name])

    def get_slice(self, name: str) -> _MetaSlice:
        return _MetaSlice(self._header[name])


def _meta_load_file(filename: Any, device: Any = "cpu") -> dict[str, torch.Tensor]:
    state = _state()
    with state.loading():
        header, _ = _read_header(state.remap(filename))
    return {name: _meta_tensor(info) for name, info in header.items()}


def _meta_load(data: bytes) -> dict[str, torch.Tensor]:
    (length,) = struct.unpack("<Q", data[:8])
    header, _ = _parse_header(data[8 : 8 + length])
    return {name: _meta_tensor(info) for name, info in header.items()}


def _patch_safetensors(patches: _Patches, module: ModuleType) -> None:
    if "safe_open" in vars(module):
        patches.set(module, "safe_open", _MetaSafeOpen)


def _patch_safetensors_torch(patches: _Patches, module: ModuleType) -> None:
    patches.set(module, "load_file", _meta_load_file)
    patches.set(module, "load", _meta_load)
    if "safe_open" in vars(module):
        patches.set(module, "safe_open", _MetaSafeOpen)


def _patch_torch_load(patches: _Patches, state: RunState) -> None:
    real = torch.load

    def load(f: Any, map_location: Any = None, *args: Any, **kwargs: Any) -> Any:
        if isinstance(f, (str, bytes, os.PathLike)):
            f = state.remap(f)
            kwargs.setdefault("mmap", True)
        with state.loading():
            return real(f, META, *args, **kwargs)

    patches.set(torch, "load", load)


def _patch_open(patches: _Patches, state: RunState) -> None:
    if not state.path_map:
        return
    for owner in (builtins, io):
        current = owner.open

        def open_mapped(file: Any, *args: Any, _open: Any = current, **kwargs: Any) -> Any:
            return _open(state.remap(file), *args, **kwargs)

        patches.set(owner, "open", open_mapped)


class _CallbackLoader(importlib.abc.Loader):
    def __init__(self, loader: Any, callback: Callable[[ModuleType], None]) -> None:
        self._loader = loader
        self._callback = callback

    def create_module(self, spec: Any) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._callback(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _ImportHook(importlib.abc.MetaPathFinder):
    """Patch a library the moment it is imported, or at once if it already is.

    Nothing is imported just to patch it: a script that never touches
    transformers never pays for importing it.
    """

    def __init__(self) -> None:
        self._callbacks: dict[str, Callable[[ModuleType], None]] = {}
        self._done: set[str] = set()
        self._finding: set[str] = set()

    def on_import(self, name: str, callback: Callable[[ModuleType], None]) -> None:
        self._callbacks[name] = callback

    def _fire(self, name: str, module: ModuleType) -> None:
        if name not in self._done:
            self._done.add(name)
            self._callbacks[name](module)

    def install(self) -> None:
        for name in list(self._callbacks):
            module = sys.modules.get(name)
            if module is not None:
                self._fire(name, module)
        sys.meta_path.insert(0, self)

    def remove(self) -> None:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self)

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname not in self._callbacks or fullname in self._done or fullname in self._finding:
            return None
        self._finding.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._finding.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _CallbackLoader(spec.loader, lambda module: self._fire(fullname, module))
        return spec


def _register_library_patches(
    hook: _ImportHook, patches: _Patches, state: RunState, hf_flags: bool
) -> None:
    hook.on_import("safetensors", lambda m: _patch_safetensors(patches, m))
    hook.on_import("safetensors.torch", lambda m: _patch_safetensors_torch(patches, m))
```

In `install()`, after `_patch_compile(patches, state)`:

```python
        _patch_torch_load(patches, state)
        _patch_open(patches, state)
        hook = _ImportHook()
        _register_library_patches(hook, patches, state, hf_flags)
        hook.install()
```

In the `finally` of `install()`, before `patches.restore()`, add `hook.remove()`. Create `hook = _ImportHook()` before the `try:` so `finally` can always reach it, and keep the `_register_library_patches`/`hook.install()` calls inside the `try:`.

- [ ] **Step 4: Run the tests**

Run: `just test tests/test_metaruntime.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
just lint
git add src/torchtyc/metaruntime.py tests/test_metaruntime.py pyproject.toml
git commit -m "feat: header-only safetensors, meta torch.load and path-map for runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: transformers: `from_pretrained` on meta, `is_tracing`, `is_torchdynamo_compiling`, the canary

**Files:**
- Modify: `src/torchtyc/metaruntime.py`
- Test: `tests/test_metaruntime.py`

**Interfaces:**
- Consumes: `_ImportHook.on_import`, `_Patches`, `RunState.loading/remap`, `_register_library_patches(..., hf_flags)`.
- Produces: `RunState.remember_config(config)`; patched `_BaseAutoModelClass.from_pretrained` and every Auto subclass that carries its own copy; patched `PreTrainedModel.from_pretrained`; `state.ignored` filled with the dropped kwargs; `state.active`/`state.inactive` entries naming the two flags.

- [ ] **Step 1: Write the failing tests** (append)

```python
from hf_models import TINY_VOCAB, requires_transformers, write_remote_checkpoint, write_tiny_llama, write_tokenizer


@requires_transformers
def test_from_pretrained_builds_on_meta(tmp_path):
    path = write_tiny_llama(tmp_path / "m")
    with metaruntime.install() as state:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0", low_cpu_mem_usage=True
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
```

If the Task 1 probe dropped `is_torchdynamo_compiling`, 4.x's sdpa path still reads the mask with the flags on. In that case, when `transformers.__version__` starts with `"4."`, the canary expects `on == {"eager": "completed", "sdpa": "run-stopped"}`. Leave the rest as written.

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_metaruntime.py -q -k "pretrained or remote or tokenizer or canary or missing_checkpoint"`
Expected: FAIL. The weights get loaded for real, or `from_pretrained` fails because the checkpoint has no weight files.

- [ ] **Step 3: Implement** (add to `metaruntime.py`)

```python
_IGNORED_KWARGS = (
    "device_map",
    "quantization_config",
    "low_cpu_mem_usage",
    "max_memory",
    "offload_folder",
    "offload_state_dict",
    "load_in_8bit",
    "load_in_4bit",
    "tp_plan",
)
_HUB_KWARGS = (
    "revision",
    "subfolder",
    "cache_dir",
    "local_files_only",
    "token",
    "force_download",
    "proxies",
    "code_revision",
)


def _transformers_major() -> int:
    version = getattr(sys.modules.get("transformers"), "__version__", "5")
    return int(version.split(".")[0])


def _as_dtype(value: Any, config: Any) -> Any:
    if value == "auto":
        value = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(value, str):
        value = getattr(torch, value, None)
    return value


def _meta_from_pretrained(*, auto: bool) -> Any:
    def from_pretrained(cls: Any, pretrained_model_name_or_path: Any, *model_args: Any, **kwargs: Any) -> Any:
        state = _state()
        path = state.remap(pretrained_model_name_or_path)
        trust = kwargs.pop("trust_remote_code", None)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = dtype if dtype is not None else torch_dtype
        attn = kwargs.pop("attn_implementation", None)
        for key in _IGNORED_KWARGS:
            if key in kwargs:
                kwargs.pop(key)
                state.ignored.add(key)
        config = kwargs.pop("config", None)
        hub = {key: kwargs.pop(key) for key in _HUB_KWARGS if key in kwargs}
        with state.loading():
            if config is None or isinstance(config, (str, os.PathLike)):
                from transformers import AutoConfig

                config, unused = AutoConfig.from_pretrained(
                    state.remap(config) if config is not None else path,
                    trust_remote_code=trust,
                    return_unused_kwargs=True,
                    **hub,
                    **kwargs,
                )
                state.ignored.update(unused)
            build: dict[str, Any] = {}
            if attn is not None:
                build["attn_implementation"] = attn
            dtype = _as_dtype(dtype, config)
            if dtype is not None:
                build["dtype" if _transformers_major() >= 5 else "torch_dtype"] = dtype
            with torch.device(META):
                if auto:
                    model = cls.from_config(config, trust_remote_code=trust, **build)
                else:
                    model = cls._from_config(config, **build)
        state.remember_config(config)
        model.eval()
        return model

    return classmethod(from_pretrained)


def _patch_auto_factory(patches: _Patches, module: ModuleType) -> None:
    patches.set(module._BaseAutoModelClass, "from_pretrained", _meta_from_pretrained(auto=True))


def _patch_auto_classes(patches: _Patches, module: ModuleType) -> None:
    """4.x copies `from_pretrained` onto every Auto class; patch each copy."""
    base = sys.modules["transformers.models.auto.auto_factory"]._BaseAutoModelClass
    for value in list(vars(module).values()):
        if isinstance(value, type) and issubclass(value, base) and "from_pretrained" in vars(value):
            patches.set(value, "from_pretrained", _meta_from_pretrained(auto=True))


def _patch_modeling_utils(patches: _Patches, module: ModuleType) -> None:
    patches.set(module.PreTrainedModel, "from_pretrained", _meta_from_pretrained(auto=False))


def _replace_everywhere(patches: _Patches, real: Any, replacement: Any) -> None:
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "transformers" or name.startswith("transformers.")):
            continue
        for attr, value in list(vars(module).items()):
            if value is real:
                patches.set(module, attr, replacement)


def _patch_import_utils(patches: _Patches, state: RunState, module: ModuleType) -> None:
    version = getattr(sys.modules.get("transformers"), "__version__", "?")
    real_tracing = vars(module).get("is_tracing")
    if real_tracing is None:
        state.inactive.append(f"transformers is_tracing (not in {version})")
    else:

        def is_tracing(tensor: Any = None) -> bool:
            return _is_meta(tensor) or real_tracing(tensor)

        _replace_everywhere(patches, real_tracing, is_tracing)
        state.active.append("transformers is_tracing (true for meta tensors)")
    real_compiling = vars(module).get("is_torchdynamo_compiling")
    if real_compiling is None:
        state.inactive.append(f"transformers is_torchdynamo_compiling (not in {version})")
    else:
        _replace_everywhere(patches, real_compiling, lambda: True)
        state.active.append("transformers is_torchdynamo_compiling (compile-safe paths)")
```

`RunState.remember_config`:

```python
    def remember_config(self, config: Any) -> None:
        values = config.to_dict() if hasattr(config, "to_dict") else {}
        self.model_config = {k: v for k, v in values.items() if type(v) is int}
```

Extend `_register_library_patches`:

```python
    hook.on_import("transformers.models.auto.auto_factory", lambda m: _patch_auto_factory(patches, m))
    hook.on_import("transformers.models.auto.modeling_auto", lambda m: _patch_auto_classes(patches, m))
    hook.on_import("transformers.modeling_utils", lambda m: _patch_modeling_utils(patches, m))
    if hf_flags:
        hook.on_import("transformers.utils.import_utils", lambda m: _patch_import_utils(patches, state, m))
```

If the Task 1 probe dropped the dynamo flag, delete the `real_compiling` branch and always append `transformers is_torchdynamo_compiling (left off; see spec §7)` to `state.inactive`.

- [ ] **Step 4: Run the tests**

Run: `just test tests/test_metaruntime.py -q`
Expected: PASS.

Then run against the other pins:
```bash
for v in 4.45.2 4.57.6; do uv run --with "transformers==$v" pytest tests/test_metaruntime.py -q || echo "FAIL $v"; done
```
Expected: PASS for both. If a version fails, the fix goes into `_meta_from_pretrained` or the callbacks, not into the tests. The tests describe behaviour every pinned version must have.

- [ ] **Step 5: Commit**

```bash
just lint
git add src/torchtyc/metaruntime.py tests/test_metaruntime.py
git commit -m "feat: build Hugging Face models on meta during a run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Worker: run jobs, a streaming child per entry, the run diagnostic

**Files:**
- Modify: `src/torchtyc/worker.py`
- Test: `tests/test_worker_run.py` (new; in-process `run_job` with a `StringIO` channel)

**Interfaces:**
- Consumes: `metaruntime.install`, `metaruntime.classify`, `Outcome`, `RunState`; `effects.active_guard`, `effects.unwrap_blocked`; `sitepaths.is_system_path`.
- Produces: the job format `{"mode": "run", "entries": [{"path": abs, "func": str|None, "args": [...], "spec": str}], "path_map": [[old, new]], "allow_effects": bool, "root": abs, "protect_roots": [abs], "stream": True}`. Events on the channel, one JSON object per line:
  - `{"event": "entry_start", "entry": spec}`
  - `{"event": "diagnostic", "diagnostic": {...}}`
  - `{"event": "heartbeat", "entry": spec, "path": str|None, "line": int|None}`
  - `{"event": "entry_result", "status": {"entry", "state", "path", "line", "reason", "active", "inactive", "ignored"}}`
  - `{"event": "done"}`

  Also `_module_name(path) -> tuple[Path, str]`, now shared with `import_from_path`.

- [ ] **Step 1: Write the failing tests** (`tests/test_worker_run.py`)

```python
"""The worker's run mode, called in-process with a StringIO channel."""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from torchtyc import worker


def run(tmp_path: Path, source: str, func: str | None = "main", args=(), files=None, path_map=()):
    for name, text in (files or {}).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(textwrap.dedent(text))
    path = tmp_path / "script.py"
    path.write_text(textwrap.dedent(source))
    spec = f"{path}::{func}" if func else str(path)
    job = {
        "mode": "run",
        "entries": [{"path": str(path), "func": func, "args": list(args), "spec": spec}],
        "path_map": [list(p) for p in path_map],
        "allow_effects": False,
        "root": str(tmp_path),
        "protect_roots": [str(tmp_path)],
        "stream": True,
    }
    channel = io.StringIO()
    worker.run_job(job, channel=channel)
    events = [json.loads(line) for line in channel.getvalue().splitlines() if line.strip()]
    diagnostics = [e["diagnostic"] for e in events if e["event"] == "diagnostic"]
    (status,) = [e["status"] for e in events if e["event"] == "entry_result"]
    return diagnostics, status, events


def line_of(source: str, needle: str) -> int:
    return next(i for i, text in enumerate(textwrap.dedent(source).splitlines()) if needle in text)


BUG = """
import torch

def project(x):
    w = torch.empty(16, 8)
    return x @ w

def main():
    project(torch.empty(4, 8))
"""


def test_shape_bug_is_a_run_error_at_the_user_line(tmp_path):
    diagnostics, status, events = run(tmp_path, BUG)
    (d,) = diagnostics
    assert d["rule"] == "run-error"
    assert d["line"] == line_of(BUG, "return x @ w")
    assert d["path"] == str(tmp_path / "script.py")
    assert "main (line" in d["via"] and "project (line" in d["via"]
    assert status["state"] == "error"
    assert events[0] == {"event": "entry_start", "entry": f"{tmp_path / 'script.py'}::main"}
    assert events[-1] == {"event": "done"}


def test_clean_run_completes(tmp_path):
    diagnostics, status, _ = run(tmp_path, "import torch\ndef main():\n    torch.empty(2, 3) @ torch.empty(3, 4)\n")
    assert diagnostics == []
    assert status["state"] == "completed"
    assert any("torch.compile" in a for a in status["active"])


def test_value_read_stops(tmp_path):
    source = "import torch\ndef main():\n    loss = torch.empty(()).sum()\n    print(f'{loss.item():.3f}')\n"
    diagnostics, status, _ = run(tmp_path, source)
    assert [d["rule"] for d in diagnostics] == ["run-stopped"]
    assert diagnostics[0]["line"] == 3
    assert status["state"] == "stopped"


def test_bug_in_helper_module_lands_there(tmp_path):
    files = {"helpers.py": "import torch\ndef bad(x):\n    return x.view(7)\n"}
    source = "import torch\nfrom helpers import bad\ndef main():\n    bad(torch.empty(3))\n"
    diagnostics, _, _ = run(tmp_path, source, files=files)
    (d,) = diagnostics
    assert d["path"] == str(tmp_path / "helpers.py")
    assert d["line"] == 2


def test_bare_file_runs_main_block_with_sibling_import(tmp_path):
    files = {"sib.py": "VALUE = 3\n"}
    source = "import torch\nimport sib\nif __name__ == '__main__':\n    torch.empty(sib.VALUE).view(2)\n"
    diagnostics, _, _ = run(tmp_path, source, func=None, files=files)
    assert [d["rule"] for d in diagnostics] == ["run-error"]


def test_package_file_runs_like_dash_m(tmp_path):
    files = {"pkg/__init__.py": "", "pkg/util.py": "N = 5\n"}
    (tmp_path / "pkg").mkdir()
    script = tmp_path / "pkg" / "tool.py"
    script.write_text("import torch\nfrom .util import N\nif __name__ == '__main__':\n    torch.empty(N).view(2)\n")
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    job = {
        "mode": "run",
        "entries": [{"path": str(script), "func": None, "args": [], "spec": str(script)}],
        "path_map": [],
        "allow_effects": False,
        "root": str(tmp_path),
        "protect_roots": [str(tmp_path)],
        "stream": True,
    }
    channel = io.StringIO()
    worker.run_job(job, channel=channel)
    events = [json.loads(line) for line in channel.getvalue().splitlines()]
    rules = [e["diagnostic"]["rule"] for e in events if e["event"] == "diagnostic"]
    assert rules == ["run-error"]  # the view fails; the relative import did not


def test_argv_reaches_argparse(tmp_path):
    source = """
    import argparse, torch
    def main():
        args = argparse.ArgumentParser()
        args.add_argument("--n", type=int)
        n = args.parse_args().n
        torch.empty(n).view(3)
    """
    diagnostics, _, _ = run(tmp_path, source, args=["--n", "4"])
    assert [d["rule"] for d in diagnostics] == ["run-error"]
    assert "4" in diagnostics[0]["message"] or "4" in (diagnostics[0]["got"] or "")


@pytest.mark.parametrize(("code", "state"), [(0, "completed"), (2, "stopped")])
def test_system_exit(tmp_path, code, state):
    _, status, _ = run(tmp_path, f"import sys\ndef main():\n    sys.exit({code})\n")
    assert status["state"] == state


def test_write_at_end_is_stopped_not_error(tmp_path):  # Review Focus 1
    source = "import json\ndef main():\n    json.dump({}, open('results.json', 'w'))\n"
    diagnostics, status, _ = run(tmp_path, source)
    assert [d["rule"] for d in diagnostics] == ["run-stopped"]
    assert status["state"] == "stopped"
    assert not (tmp_path / "results.json").exists()


def test_async_main_is_awaited(tmp_path):  # Review Focus 2
    source = "import torch\nasync def main():\n    torch.empty(3).view(2)\n"
    diagnostics, _, _ = run(tmp_path, source)
    assert [d["rule"] for d in diagnostics] == ["run-error"]


def test_syntax_error_anchors_to_its_line(tmp_path):  # Review Focus 3
    source = "def main():\n    pass\n\nx = (1,\n"
    diagnostics, _, _ = run(tmp_path, source)
    (d,) = diagnostics
    assert d["rule"] == "run-error"
    assert d["line"] >= 3


def test_prints_do_not_corrupt_the_channel(tmp_path):  # Review Focus 4
    source = "def main():\n    for i in range(2000):\n        print('x' * 80)\n"
    _, status, _ = run(tmp_path, source)
    assert status["state"] == "completed"


def test_model_output_hint(tmp_path):
    source = """
    import torch
    from dataclasses import dataclass

    class ModelOutput(dict):
        pass

    class CausalLMOutput(ModelOutput):
        logits = None

    def main():
        out = CausalLMOutput()
        out.logits = torch.empty(1, 5, 9)
        out.view(3)
    """
    diagnostics, _, _ = run(tmp_path, source)
    (d,) = diagnostics
    assert "`out` is a CausalLMOutput, not a tensor" in d["hint"]
    assert "out.logits" in d["hint"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_worker_run.py -q`
Expected: FAIL. `run_job` ignores `mode` and looks for `job["paths"]`, so it raises `KeyError: 'paths'`.

- [ ] **Step 3: Refactor `import_from_path` onto `_module_name`** (in `worker.py`)

```python
def _module_name(path: Path) -> tuple[Path, str]:
    """The sys.path root and dotted name a file imports under, packages included."""
    path = path.resolve()
    root = path.parent
    # `pkg/__init__.py` is the module `pkg`, not `pkg.__init__`.
    parts = [] if path.stem == "__init__" else [path.stem]
    while (root / "__init__.py").exists():
        parts.append(root.name)
        root = root.parent
    return root, ".".join(reversed(parts))
```

In `import_from_path`, replace the inline computation (`root = path.parent` through `dotted = ...`) with `root, dotted = _module_name(path)` and keep the rest unchanged.

Run: `just test tests/test_engine.py -q`
Expected: PASS (the refactor doesn't change behaviour).

- [ ] **Step 4: Implement run mode** (add to `worker.py`; new top-level imports: `asyncio`, `inspect`, `io` if not present)

```python
_TORCHTYC_DIR = Path(__file__).resolve().parent
_HEARTBEAT_SECONDS = 1.0
_EINOPS_RANK = re.compile(r"expected (\d+) dims\. Received (\d+)-dim tensor")


def _is_project_file(filename: str | None, root: Path) -> bool:
    if not filename or filename.startswith("<"):
        return False
    try:
        path = Path(filename).resolve()
    except (OSError, ValueError):
        return False
    if not path.is_relative_to(root) or path.is_relative_to(_TORCHTYC_DIR):
        return False
    return not is_system_path(path)


def _deepest_project_line(frame: Any, root: Path) -> tuple[str, int] | None:
    while frame is not None:
        if _is_project_file(frame.f_code.co_filename, root):
            return frame.f_code.co_filename, frame.f_lineno - 1
        frame = frame.f_back
    return None


def _run_as_main(path: Path) -> None:
    import runpy

    top, dotted = _module_name(path)
    if "." in dotted:
        if str(top) not in sys.path:
            sys.path.insert(0, str(top))
        runpy.run_module(dotted, run_name="__main__", alter_sys=True)
    else:
        sys.path.insert(0, str(path.parent))
        runpy.run_path(str(path), run_name="__main__")


def _names_on_line(statement: str | None) -> list[str]:
    if not statement:
        return []
    try:
        tree = ast.parse(statement.strip())
    except SyntaxError:
        return []
    seen: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in seen:
            seen.append(node.id)
    return seen


def _is_model_output(value: Any) -> bool:
    return any(cls.__name__ == "ModelOutput" for cls in type(value).__mro__)


def _run_got(frame: Any, statement: str | None, state: Any) -> str | None:
    parts: list[str] = []
    for name in _names_on_line(statement):
        value = frame.f_locals.get(name) if frame is not None else None
        if isinstance(value, torch.Tensor):
            parts.append(f"{name} {state.render_shape(tuple(value.shape))}")
        elif value is not None and _is_model_output(value):
            parts.append(f"{name} {type(value).__name__}")
    return "; ".join(parts) or None


def _run_hint(exc: BaseException, frame: Any, statement: str | None) -> str | None:
    for name in _names_on_line(statement):
        value = frame.f_locals.get(name) if frame is not None else None
        if value is not None and _is_model_output(value):
            hint = f"`{name}` is a {type(value).__name__}, not a tensor"
            if getattr(value, "logits", None) is not None:
                hint += f"; did you mean `{name}.logits`?"
            return hint
    match = _EINOPS_RANK.search(str(exc))
    if match:
        return f"the pattern names {match[1]} axes but the tensor has {match[2]}"
    return None


def _callee(frames: list[traceback.FrameSummary], index: int) -> str | None:
    """The function the user's line called, skipping torchtyc's own wrappers.

    The einops recorder sits between the user's `rearrange(...)` and einops,
    and its frame must not name the call.
    """
    for summary in frames[index + 1 :]:
        try:
            if Path(summary.filename).resolve().is_relative_to(_TORCHTYC_DIR):
                continue
        except (OSError, ValueError):
            pass
        if summary.name.startswith("_") or summary.name == "<module>":
            return None
        return summary.name
    return None


def _run_diagnostic(exc: BaseException, outcome: Any, entry_path: str, root: Path, state: Any) -> Diagnostic:
    summaries = traceback.extract_tb(exc.__traceback__)
    frames = _live_frames(exc)
    mine = [i for i, s in enumerate(summaries) if _is_project_file(s.filename, root)]
    path, line, column, end_line, end_column = entry_path, 0, 0, 0, 1
    frame = statement = None
    if isinstance(exc, SyntaxError) and exc.filename and _is_project_file(exc.filename, root):
        path, line = exc.filename, max((exc.lineno or 1) - 1, 0)
        column, end_line, end_column = max((exc.offset or 1) - 1, 0), line, max(exc.offset or 1, 1)
    elif mine:
        chosen = summaries[mine[-1]]
        frame, statement = frames[mine[-1]], chosen.line
        path, line = chosen.filename, chosen.lineno - 1
        end_line = (chosen.end_lineno or chosen.lineno) - 1
        if chosen.colno is not None and chosen.end_colno is not None:
            column, end_column = chosen.colno, chosen.end_colno
        else:
            end_column = len(chosen.line or "") or 1
    via = " → ".join(f"{summaries[i].name} (line {summaries[i].lineno})" for i in mine) or None
    if outcome.rule == "run-error":
        last = (str(exc).strip().splitlines() or [""])[-1]
        callee = _callee(summaries, mine[-1]) if mine else None
        head = f"`{callee}` raised" if callee else "raised"
        message = f"{head} {type(exc).__name__}: {last}" if last else f"{head} {type(exc).__name__}"
    else:
        message = outcome.reason
    return Diagnostic(
        path=path,
        line=line,
        column=column,
        end_line=end_line,
        end_column=end_column,
        rule=outcome.rule,
        severity=_severity(outcome.rule),
        message=message,
        got=_run_got(frame, statement, state) if outcome.rule == "run-error" else None,
        hint=_run_hint(exc, frame, statement) if outcome.rule == "run-error" else None,
        via=via,
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def run_entry(entry: dict[str, Any], job: dict[str, Any], emit: Any) -> dict[str, Any]:
    """Run one entry point on meta, emitting diagnostics as they are found."""
    _ensure_initialized()
    from . import metaruntime

    path = entry["path"]
    root = Path(job["root"]).resolve()
    status: dict[str, Any] = {
        "entry": entry["spec"],
        "state": "completed",
        "path": None,
        "line": None,
        "reason": None,
        "active": [],
        "inactive": [],
        "ignored": [],
    }
    saved_argv, saved_path = list(sys.argv), list(sys.path)
    main_ident = threading.get_ident()
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(_HEARTBEAT_SECONDS):
            where = _deepest_project_line(sys._current_frames().get(main_ident), root)
            emit(
                {
                    "event": "heartbeat",
                    "entry": entry["spec"],
                    "path": where[0] if where else None,
                    "line": where[1] if where else None,
                }
            )

    heartbeat = threading.Thread(target=beat, name="torchtyc-heartbeat", daemon=True)
    try:
        with active_guard(enabled=False, protect=job.get("protect_roots") or [str(root)]):
            with active_guard(enabled=not job.get("allow_effects", False), phase="run"):
                with metaruntime.install(job.get("path_map", ())) as state:
                    heartbeat.start()
                    try:
                        sys.argv = [path, *entry.get("args", [])]
                        if entry.get("func"):
                            module = import_from_path(Path(path))
                            result = getattr(module, entry["func"])()
                            if inspect.iscoroutine(result):
                                asyncio.run(result)
                        else:
                            _run_as_main(Path(path))
                    except KeyboardInterrupt:
                        raise
                    except BaseException as exc:  # noqa: BLE001
                        outcome = metaruntime.classify(exc)
                        status["state"] = outcome.state
                        if outcome.rule is not None:
                            diagnostic = _run_diagnostic(exc, outcome, path, root, state)
                            emit({"event": "diagnostic", "diagnostic": diagnostic.to_json()})
                            status.update(
                                path=diagnostic.path,
                                line=diagnostic.line,
                                reason=outcome.reason or None,
                            )
                    status.update(
                        active=list(state.active),
                        inactive=list(state.inactive),
                        ignored=sorted(state.ignored),
                    )
    finally:
        stop.set()
        sys.argv = saved_argv
        sys.path[:] = saved_path
    return status


def _run_entry_in_child(entry: dict[str, Any], job: dict[str, Any], channel: Any) -> None:
    r_fd, w_fd = os.pipe()
    for stream in (sys.stdout, sys.stderr, channel):
        if stream is not None:
            stream.flush()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        lock = threading.Lock()

        def emit(message: dict[str, Any]) -> None:
            data = memoryview((json.dumps(message) + "\n").encode("utf-8"))
            with lock:
                while data:
                    data = data[os.write(w_fd, data) :]

        code = 0
        try:
            emit({"event": "entry_result", "status": run_entry(entry, job, emit)})
        except KeyboardInterrupt:
            code = 130
        except BaseException as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                emit(
                    {
                        "event": "entry_result",
                        "status": {"entry": entry["spec"], "state": "crashed", "reason": f"{type(exc).__name__}: {exc}"},
                    }
                )
        finally:
            with contextlib.suppress(OSError):
                os.close(w_fd)
            os._exit(code)

    os.close(w_fd)
    finished = False
    with os.fdopen(r_fd, "r", encoding="utf-8") as stream:
        for line in stream:
            channel.write(line)
            channel.flush()
            finished = finished or '"entry_result"' in line
    _, wait_status = os.waitpid(pid, 0)
    if not finished:
        detail = (
            f"signal {os.WTERMSIG(wait_status)}" if os.WIFSIGNALED(wait_status) else f"exit code {os.WEXITSTATUS(wait_status)}"
        )
        diagnostic = Diagnostic(
            path=entry["path"],
            line=0,
            column=0,
            rule="worker-error",
            severity=_severity("worker-error"),
            message=f"the run's child process died ({detail})",
        )
        channel.write(json.dumps({"event": "diagnostic", "diagnostic": diagnostic.to_json()}) + "\n")
        channel.write(
            json.dumps({"event": "entry_result", "status": {"entry": entry["spec"], "state": "crashed", "reason": detail}})
            + "\n"
        )
        channel.flush()


def _run_entries(job: dict[str, Any], channel: Any) -> dict[str, Any]:
    _ensure_initialized()
    channel = channel if channel is not None else sys.stderr
    lock = threading.Lock()

    def emit(message: dict[str, Any]) -> None:
        with lock:
            channel.write(json.dumps(message) + "\n")
            channel.flush()

    for entry in job["entries"]:
        emit({"event": "entry_start", "entry": entry["spec"]})
        if hasattr(os, "fork"):
            _run_entry_in_child(entry, job, channel)
        else:
            emit({"event": "entry_result", "status": run_entry(entry, job, emit)})
    emit({"event": "done"})
    return {"diagnostics": [], "hovers": {}, "checked_functions": 0, "skipped_functions": 0}
```

At the top of `run_job`:

```python
    if job.get("mode") == "run":
        return _run_entries(job, channel)
```

Two remaining details:
- `state.render_shape` comes in Task 10. Until then, add this temporary method to `RunState` so Task 8 stays green; Task 10 replaces it:
  ```python
  def render_shape(self, shape: tuple[int, ...]) -> str:
      return "(" + ", ".join(str(s) for s in shape) + ")"
  ```
- The syntax-error case: `import_from_path` raises `SyntaxError` from the compile. `classify` returns `run-error` for it, and `_run_diagnostic` anchors it to `exc.lineno`.

- [ ] **Step 5: Run the tests**

Run: `just test tests/test_worker_run.py tests/test_engine.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
just lint
git add src/torchtyc/worker.py src/torchtyc/metaruntime.py tests/test_worker_run.py
git commit -m "feat: run entry points on meta in the worker, streaming findings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Engine: the run phase of `check_paths`

**Files:**
- Modify: `src/torchtyc/engine.py`
- Test: `tests/test_run.py` (new; end-to-end through a real worker subprocess)

**Interfaces:**
- Consumes: the worker's run job and events (Task 8), `RunEntry` and `Config.run_timeout`/`path_map` (Task 2).
- Produces:
  - `RunStatus` (the fields `entry, state, path, line, reason, active, inactive, ignored`) with `RunStatus.from_json(data)`;
  - `Report.run_status: list[RunStatus]`;
  - `EntryError(ValueError)` and `entry_problem(entry: RunEntry) -> Diagnostic | None` (raises `EntryError`);
  - `_run_worker_entries(entries, config, *, cancel=None, on_proc=None) -> tuple[list[Diagnostic], list[RunStatus], str | None]`;
  - the new `check_paths(..., run: Sequence[RunEntry] = (), trace: bool = True)`;
  - a new `_communicate(..., reset_events=("file_start", "file_result"))` parameter.

- [ ] **Step 1: Write the failing tests** (`tests/test_run.py`)

```python
"""`torchtyc run` end to end: a real project in tmp_path, a real worker."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from torchtyc.config import Config, RunEntry
from torchtyc.engine import EntryError, check_paths


@pytest.fixture
def project(tmp_path: Path):
    def build(source: str, name: str = "script.py", **config_kwargs) -> tuple[Path, Config]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
        return path, Config(root=tmp_path, python=sys.executable, **config_kwargs)

    return build


def run(path: Path, config: Config, func: str | None = "main", args=()):
    entry = RunEntry(str(path), func, tuple(args))
    return check_paths([str(path)], config, run=[entry], trace=False)


BUG = """
import torch

def main():
    x = torch.empty(4, 8)
    x @ torch.empty(16, 8)
"""


def test_run_error_reported_with_status(project):
    path, config = project(BUG)
    report = run(path, config)
    assert [d.rule for d in report.diagnostics] == ["run-error"]
    assert report.diagnostics[0].line == 5
    assert report.diagnostics[0].path == str(path)
    (status,) = report.run_status
    assert (status.state, status.line) == ("error", 5)
    assert report.errors == 1


def test_missing_function_is_an_entry_error(project):
    path, config = project(BUG)
    with pytest.raises(EntryError, match="no function `mainn`.*main"):
        run(path, config, func="mainn")


def test_required_parameters_are_unresolved_arg(project):
    path, config = project("def main(ckpt):\n    pass\n")
    report = run(path, config)
    assert [d.rule for d in report.diagnostics] == ["unresolved-arg"]
    assert report.run_status[0].state == "skipped"


def test_timeout_keeps_earlier_findings(project):
    # A finding ends its own run, so "earlier findings" are those of earlier
    # entries: the first entry's error must survive the second one's timeout.
    first, config = project(BUG, name="first.py", run_timeout=2.0)
    second, _ = project("import time\n\ndef main():\n    while True:\n        time.sleep(0.05)\n", name="second.py")
    entries = [RunEntry(str(first), "main"), RunEntry(str(second), "main")]
    report = check_paths([str(first), str(second)], config, run=entries, trace=False)
    rules = {(Path(d.path).name, d.rule) for d in report.diagnostics}
    assert rules == {("first.py", "run-error"), ("second.py", "run-stopped")}
    (stopped,) = [d for d in report.diagnostics if d.rule == "run-stopped"]
    assert "timed out after 2s" in stopped.message
    assert stopped.line in (3, 4)  # inside the loop, from the last heartbeat
    assert [s.state for s in report.run_status] == ["error", "timeout"]


def test_suppression_in_helper_file(project, tmp_path):
    (tmp_path / "helpers.py").write_text(
        "import torch\ndef bad(x):\n    return x.view(7)  # torchtyc: ignore[run-error]\n"
    )
    path, config = project("import torch\nfrom helpers import bad\ndef main():\n    bad(torch.empty(3))\n")
    report = run(path, config)
    assert report.diagnostics == []


def test_suppression_in_entry_file_is_used(project):
    path, config = project("import torch\ndef main():\n    torch.empty(3).view(7)  # torchtyc: ignore[run-error]\n")
    report = run(path, config)
    assert [d.rule for d in report.diagnostics] == []


def test_path_map_end_to_end(project, tmp_path):
    from hf_models import write_inputs

    (tmp_path / "data").mkdir()
    write_inputs(tmp_path / "data" / "inputs.safetensors", {"a": 6})
    source = """
    from safetensors.torch import load_file

    def main():
        ids = load_file("/workspace/data/inputs.safetensors")["a"]
        ids.view(4)
    """
    path, config = project(source, path_map=(("/workspace", str(tmp_path)),))
    report = run(path, config)
    assert [d.rule for d in report.diagnostics] == ["run-error"]
    assert "6" in (report.diagnostics[0].got or "") or "6" in report.diagnostics[0].message


def test_missing_input_through_open_is_stopped(project):
    path, config = project("def main():\n    open('/definitely/not/here.json').read()\n")
    report = run(path, config)
    assert [d.rule for d in report.diagnostics] == ["run-stopped"]


def test_check_runs_config_entries_inside_checked_paths(project, tmp_path):
    path, config = project(BUG, name="tools/script.py")
    config.run = (RunEntry(str(path), "main"),)
    report = check_paths([str(tmp_path / "tools" / "script.py")], config, run=config.run)
    assert "run-error" in [d.rule for d in report.diagnostics]


def test_run_error_on_trace_error_line_is_dropped(project):
    source = """
    import torch
    from jaxtyping import Float
    from torch import Tensor

    def f(x: Float[Tensor, "n"]) -> Float[Tensor, "n"]:
        return x.view(7)

    def main():
        f(torch.empty(3))
    """
    path, config = project(source)
    report = check_paths([str(path)], config, run=[RunEntry(str(path), "main")])
    rules = [d.rule for d in report.diagnostics if d.line == 6]
    assert rules == ["trace-error"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_run.py -q`
Expected: FAIL at import: `EntryError` doesn't exist, and `check_paths` doesn't accept `run=`.

- [ ] **Step 3: Implement** (in `engine.py`)

Add `import contextlib` and `from collections.abc import Sequence` to the imports, and `from .config import Config, RunEntry`.

```python
@dataclass
class RunStatus:
    """How one `torchtyc run` entry ended."""

    entry: str
    state: str
    path: str | None = None
    line: int | None = None
    reason: str | None = None
    active: list[str] = field(default_factory=list)
    inactive: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict) -> RunStatus:
        return cls(
            entry=data.get("entry", "?"),
            state=data.get("state", "crashed"),
            path=data.get("path"),
            line=data.get("line"),
            reason=data.get("reason"),
            active=list(data.get("active") or []),
            inactive=list(data.get("inactive") or []),
            ignored=list(data.get("ignored") or []),
        )


class EntryError(ValueError):
    """A run entry that names a file or function that is not there."""


def entry_problem(entry: RunEntry) -> Diagnostic | None:
    """Check an entry from source, without importing anything.

    A missing file or function is the user's command-line mistake (EntryError,
    exit 2). Required parameters with no default mean torchtyc cannot call it.
    """
    path = Path(entry.path)
    if not path.is_file():
        raise EntryError(f"{entry.path}: no such file")
    if entry.func is None:
        return None
    scan = scan_source(path.read_text(encoding="utf-8"), entry.path)
    if scan.syntax_error is not None:
        return None  # the run reports it, at its line, as the import fails
    top_level = [t for t in scan.targets if t.owner is None and "." not in t.qualname]
    target = next((t for t in top_level if t.qualname == entry.func), None)
    if target is None:
        available = ", ".join(t.qualname for t in top_level) or "none"
        raise EntryError(f"{entry.path} has no function `{entry.func}` (functions here: {available})")
    required = [p.name for p in target.params if not p.has_default]
    if not required:
        return None
    names = ", ".join(f"`{name}`" for name in required)
    return Diagnostic(
        path=entry.path,
        line=target.position.line,
        column=target.position.column,
        end_line=target.position.end_line,
        end_column=target.position.end_column,
        rule="unresolved-arg",
        severity=RULES["unresolved-arg"].severity,
        message=f"`{entry.func}` needs {names} and has no defaults, so torchtyc cannot call it",
        function=entry.func,
    )
```

In `Report`, add `run_status: list[RunStatus] = field(default_factory=list)`. Define `RunStatus` above `Report`.

In `_communicate`, add the parameter `reset_events: tuple[str, ...] = ("file_start", "file_result")` and change `elif event in ("file_start", "file_result"):` to `elif event in reset_events:`.

Split process creation out of `run_worker`, so both phases share it:

```python
def _spawn_worker(config: Config) -> subprocess.Popen[str]:
    env = dict(os.environ)
    if config.extra_paths:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([p for p in [*config.extra_paths, existing] if p])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.Popen(
        [config.interpreter, "-c", _BOOTSTRAP, str(_PACKAGE_ROOT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(config.root),
        env=env,
        start_new_session=(os.name == "posix"),
    )
```

Use it inside `run_worker` (replacing the inline `env` and `Popen` code, keeping the `except OSError` around the call).

```python
def _run_worker_entries(
    entries: Sequence[RunEntry],
    config: Config,
    *,
    cancel: threading.Event | None = None,
    on_proc: Callable[[subprocess.Popen[str]], None] | None = None,
) -> tuple[list[Diagnostic], list[RunStatus], str | None]:
    by_spec = {entry.spec: entry for entry in entries}
    job = {
        "mode": "run",
        "entries": [
            {"path": str(Path(e.path).resolve()), "func": e.func, "args": list(e.args), "spec": e.spec}
            for e in entries
        ],
        "path_map": [list(pair) for pair in config.path_map],
        "allow_effects": config.allow_effects,
        "root": str(Path(config.root).resolve()),
        "protect_roots": [str(Path(config.root).resolve())],
        "stream": True,
    }
    try:
        proc = _spawn_worker(config)
    except OSError as exc:
        return [], [], f"could not start {config.interpreter}: {exc}"
    if on_proc is not None:
        on_proc(proc)
    # A run's limit is wall-clock per entry: only a new entry resets it, never
    # a heartbeat, so a hung script still times out.
    lines, stderr, timed_out = _communicate(
        proc, json.dumps(job), config.run_timeout, cancel, reset_events=("entry_start",)
    )
    if not timed_out and proc.poll() is None:
        _finish(proc)

    diagnostics: list[Diagnostic] = []
    statuses: list[RunStatus] = []
    beats: dict[str, tuple[str | None, int | None]] = {}
    current: str | None = None
    error: str | None = None
    for line in lines:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = msg.get("event")
        if event == "entry_start":
            current = msg.get("entry")
        elif event == "diagnostic":
            diagnostics.append(Diagnostic.from_json(msg["diagnostic"]))
        elif event == "heartbeat":
            beats[msg.get("entry")] = (msg.get("path"), msg.get("line"))
        elif event == "entry_result":
            statuses.append(RunStatus.from_json(msg["status"]))
        elif "error" in msg:
            error = msg["error"]

    finished = {status.entry for status in statuses}
    if timed_out and current is not None and current not in finished:
        path, beat_line = beats.get(current, (None, None))
        where = f" at line {beat_line + 1}" if beat_line is not None else ""
        message = f"the run timed out after {config.run_timeout:g}s{where}"
        entry_path = by_spec[current].path if current in by_spec else (path or "")
        diagnostics.append(
            Diagnostic(
                path=path or entry_path,
                line=beat_line or 0,
                column=0,
                rule="run-stopped",
                severity=RULES["run-stopped"].severity,
                message=message,
            )
        )
        statuses.append(RunStatus(current, "timeout", path or entry_path, beat_line, message))
    elif not lines and error is None:
        detail = (stderr or "").strip().splitlines()
        error = f"the worker produced no output ({detail[-1] if detail else proc.returncode})"
    return diagnostics, statuses, error
```

In `check_paths`, change the signature to add `run: Sequence[RunEntry] = ()` and `trace: bool = True` (keyword-only). Wrap the trace call as `if trace and traceable:`.

`missing-annotation` is advice about the annotation workflow, and in `torchtyc run` it would print one info line per function in the script. Drop it when the trace pass is off. In the lint loop, replace `report.diagnostics.extend(lint_scan(scan, config))` with:

```python
        found = lint_scan(scan, config)
        if not trace:
            found = [d for d in found if d.rule != "missing-annotation"]
        report.diagnostics.extend(found)
```

`test_required_parameters_are_unresolved_arg` and `test_suppression_in_entry_file_is_used` assert exact rule lists, so they fail if this filter is missing.

Then, before `per_file` is built:

```python
    extra_scans: dict[str, FileScan] = {}
    if run:
        checked = [Path(p).resolve() for p in paths]
        selected = [
            e for e in run
            if not checked or any(Path(e.path).resolve().is_relative_to(c) or Path(e.path).resolve() == c for c in checked)
        ]
        runnable: list[RunEntry] = []
        for entry in selected:
            problem = entry_problem(entry)
            if problem is not None:
                report.diagnostics.append(problem)
                report.run_status.append(RunStatus(entry.spec, "skipped", problem.path, problem.line, problem.message))
            else:
                runnable.append(entry)
        if runnable:
            found, statuses, error = _run_worker_entries(runnable, config, cancel=cancel, on_proc=on_proc)
            # Diagnostics come back absolute; key them like the caller's paths
            # so the entry file's suppressions see them.
            absolute = {str(Path(p).resolve()): p for p in [*paths, *(e.path for e in runnable)]}
            for diagnostic in found:
                diagnostic.path = absolute.get(str(Path(diagnostic.path).resolve()), diagnostic.path)
            report.diagnostics.extend(found)
            report.run_status.extend(statuses)
            report.worker_error = report.worker_error or error
        traced_lines = {
            (str(Path(d.path).resolve()), d.line) for d in report.diagnostics if d.rule == "trace-error"
        }
        report.diagnostics = [
            d
            for d in report.diagnostics
            if not (d.rule == "run-error" and (str(Path(d.path).resolve()), d.line) in traced_lines)
        ]
        for diagnostic in report.diagnostics:
            if diagnostic.path not in scans and diagnostic.path.endswith(".py"):
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    text = Path(diagnostic.path).read_text(encoding="utf-8")
                    extra_scans[diagnostic.path] = scan_source(text, diagnostic.path)
```

In the finalization loop, use both scans; only the checked files report unused suppressions:

```python
    for path in {*per_file, *scans, *extra_scans}:
        items = per_file.get(path, [])
        scan = scans.get(path) or extra_scans.get(path)
        if scan is None:
            final.extend(items)
            continue
        final.extend(apply_suppressions(items, scan.suppressions))
        if path not in scans:
            continue
        for suppression in scan.suppressions:
            ...  # unchanged
```

`selected` with empty `paths` (the `run` command always passes the entry file, but keep it robust) runs every entry.

- [ ] **Step 4: Run the tests**

Run: `just test tests/test_run.py tests/test_engine.py -q`
Expected: PASS. `test_timeout_keeps_earlier_findings` takes about 3s.

- [ ] **Step 5: Commit**

```bash
just lint
git add src/torchtyc/engine.py tests/test_run.py
git commit -m "feat: run phase in check_paths with a hard per-entry deadline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Number labels and the einops axis recorder

**Files:**
- Modify: `src/torchtyc/metaruntime.py`
- Test: `tests/test_metaruntime.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `RunState.model_config` (Task 7), the `_ImportHook` (Task 6).
- Produces: `RunState.bind_einops(pattern: str, shape: tuple[int, ...])`, `RunState.label(size: int) -> str` and `RunState.render_shape(shape) -> str`, which replaces the Task 8 stand-in. Patched `einops.rearrange/reduce/repeat` record their axes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metaruntime.py`:

```python
def test_labels():
    state = metaruntime.RunState()
    state.model_config = {"vocab_size": 250624, "hidden_size": 2560, "intermediate_size": 2560 * 4, "num_kv": 8}
    state.bind_einops("seq_len -> 1 seq_len", (113,))
    assert state.label(250624) == "250624 (vocab_size?)"
    assert state.label(113) == "113 (seq_len?)"
    assert state.label(8) == "8"  # config labels need >= 128
    assert state.render_shape((1, 113, 250624)) == "(1, 113 (seq_len?), 250624 (vocab_size?))"
    state.model_config["other"] = 2560
    assert state.label(2560) == "2560"  # two config keys: ambiguous, left bare
    assert state.label(226) == "226"  # never factored into seq_len*2


def test_einops_axis_with_two_sizes_is_not_a_label():
    state = metaruntime.RunState()
    state.bind_einops("b n -> n b", (2, 113))
    state.bind_einops("b n -> n b", (2, 226))
    assert state.label(113) == "113"


def test_einops_calls_are_recorded():
    with metaruntime.install() as state:
        from einops import rearrange

        rearrange(torch.empty(113), "seq_len -> 1 seq_len")
    assert state.einops_axes == {"seq_len": {113}}
```

Append to `tests/test_run.py`:

```python
def test_got_line_names_axes(project):
    source = """
    import torch
    from einops import rearrange

    def main():
        ids = rearrange(torch.zeros(113, dtype=torch.long), "seq_len -> 1 seq_len")
        logits = torch.empty(1, 113, 1000)
        rearrange(logits, "1 seq_len -> seq_len")
    """
    path, config = project(source)
    report = run(path, config)
    (d,) = report.diagnostics
    assert d.got == "logits (1, 113 (seq_len?), 1000)"
    assert d.hint == "the pattern names 2 axes but the tensor has 3"
```

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_metaruntime.py tests/test_run.py -q -k "label or einops or got_line"`
Expected: FAIL, because `bind_einops` doesn't exist.

- [ ] **Step 3: Implement**

In `RunState`, replace the stand-in `render_shape` with:

```python
    def bind_einops(self, pattern: str, shape: tuple[int, ...]) -> None:
        left = pattern.split("->", 1)[0].split()
        if len(left) != len(shape) or not all(axis.isidentifier() for axis in left):
            return
        for axis, size in zip(left, shape):
            self.einops_axes.setdefault(axis, set()).add(int(size))

    def label(self, size: int) -> str:
        """A hedged name for a size, only when exactly one source gives one.

        The number is always printed, so a wrong label cannot hide the value.
        """
        names: set[str] = set()
        if size >= 128:
            keys = [key for key, value in self.model_config.items() if value == size]
            if len(keys) > 1:
                return str(size)
            names.update(keys)
        axes = [axis for axis, sizes in self.einops_axes.items() if sizes == {size}]
        if len(axes) > 1:
            return str(size)
        names.update(axes)
        return f"{size} ({names.pop()}?)" if len(names) == 1 else str(size)

    def render_shape(self, shape: tuple[int, ...]) -> str:
        return "(" + ", ".join(self.label(int(size)) for size in shape) + ")"
```

Add the einops patch (top-level `import functools`):

```python
def _patch_einops(patches: _Patches, state: RunState, module: ModuleType) -> None:
    for name in ("rearrange", "reduce", "repeat"):
        real = vars(module).get(name)
        if real is None:
            continue

        @functools.wraps(real)
        def recording(tensor: Any, pattern: Any, *args: Any, _real: Any = real, **kwargs: Any) -> Any:
            if isinstance(tensor, torch.Tensor) and isinstance(pattern, str):
                state.bind_einops(pattern, tuple(tensor.shape))
            return _real(tensor, pattern, *args, **kwargs)

        patches.set(module, name, recording)
```

and register it in `_register_library_patches`:

```python
    hook.on_import("einops", lambda m: _patch_einops(patches, state, m))
```

- [ ] **Step 4: Run the tests**

Run: `just test tests/test_metaruntime.py tests/test_run.py tests/test_worker_run.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
just lint
git add src/torchtyc/metaruntime.py tests/test_metaruntime.py tests/test_run.py
git commit -m "feat: hedged dimension labels in run diagnostics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: CLI `run`, config entries for `check`, status lines in every format

**Files:**
- Modify: `src/torchtyc/cli.py`, `src/torchtyc/formats.py`
- Test: `tests/test_run.py`, `tests/test_formats.py`

**Interfaces:**
- Consumes: `check_paths(..., run=, trace=)`, `EntryError`, `RunStatus`, `RunEntry.parse`, `Overrides.run_timeout`.
- Produces: `torchtyc run TARGET [--format ...] [common flags] [-- ARGS...]`, which exits 0, 1 or 2 like `check`; `torchtyc check` running config entries; status lines in `full`/`concise`; `run_status` in `json`; `::notice` lines in `github`; `via` in `full` and `github`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run.py`:

```python
from torchtyc.cli import main as cli_main


def test_cli_run_exit_codes_and_output(project, capsys, monkeypatch, tmp_path):
    path, _ = project(BUG, name="tools/script.py")
    (tmp_path / "pyproject.toml").write_text(f'[tool.torchtyc]\npython = "{sys.executable}"\n')
    monkeypatch.chdir(tmp_path / "tools")  # Review Focus 5: run from a subdirectory
    code = cli_main(["run", "script.py::main", "--format", "full"])
    out = capsys.readouterr().out
    assert code == 1
    assert "tools/script.py:6:" in out
    assert "error[run-error]" in out
    assert "run tools/script.py::main: stopped at tools/script.py:6 (run-error)" in out
    assert "via: main (line 6)" in out


def test_cli_run_missing_function_exits_2(project, capsys, tmp_path):
    path, _ = project(BUG)
    (tmp_path / "pyproject.toml").write_text(f'[tool.torchtyc]\npython = "{sys.executable}"\n')
    assert cli_main(["run", f"{path}::nope"]) == 2
    assert "no function `nope`" in capsys.readouterr().err


def test_cli_run_passthrough_args(project, capsys, tmp_path):
    source = """
    import argparse, torch
    def main():
        p = argparse.ArgumentParser(); p.add_argument("--n", type=int)
        torch.empty(p.parse_args().n).view(3)
    """
    path, _ = project(source)
    (tmp_path / "pyproject.toml").write_text(f'[tool.torchtyc]\npython = "{sys.executable}"\n')
    assert cli_main(["run", f"{path}::main", "--", "--n", "3"]) == 0
    assert cli_main(["run", f"{path}::main", "--", "--n", "4"]) == 1


def test_cli_run_timeout_flag(project, capsys, tmp_path):
    path, _ = project("import time\ndef main():\n    while True:\n        time.sleep(0.05)\n")
    (tmp_path / "pyproject.toml").write_text(f'[tool.torchtyc]\npython = "{sys.executable}"\n')
    assert cli_main(["run", f"{path}::main", "--timeout", "2", "--format", "concise"]) == 0
    assert "timed out after 2s" in capsys.readouterr().out


def test_cli_check_runs_config_entries(project, capsys, tmp_path):
    path, _ = project(BUG, name="tools/script.py")
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.torchtyc]\npython = "{sys.executable}"\nrun = ["tools/script.py::main"]\n'
    )
    assert cli_main(["check", str(tmp_path / "tools"), "--format", "github"]) == 1
    out = capsys.readouterr().out
    assert "title=torchtyc[run-error]" in out
    assert cli_main(["check", str(tmp_path / "elsewhere.py")]) in (0, 2)  # entry not under the path: not run
```

Append to `tests/test_formats.py`:

```python
def test_run_status_rendering(tmp_path):
    import json as json_module

    from torchtyc.engine import Report, RunStatus
    from torchtyc.formats import render

    report = Report(checked_files=1)
    report.run_status = [
        RunStatus("a.py::main", "stopped", str(tmp_path / "a.py"), 35, "`.item()` needs a tensor's value", inactive=["x"], ignored=["device_map"]),
        RunStatus("b.py::main", "completed"),
    ]
    full = render(report, "full", tmp_path, color=False)
    assert "run a.py::main: stopped at a.py:36 (`.item()` needs a tensor's value)" in full
    assert "  inactive: x" in full and "  ignored: device_map" in full
    assert "run b.py::main: ran to completion" in full
    data = json_module.loads(render(report, "json", tmp_path))
    assert data["run_status"][0]["state"] == "stopped"
    github = render(report, "github", tmp_path)
    assert "::notice title=torchtyc run::run a.py::main: stopped" in github
    assert "b.py::main" not in github
```

- [ ] **Step 2: Run them to see them fail**

Run: `just test tests/test_run.py tests/test_formats.py -q -k "cli or run_status"`
Expected: FAIL, because there's no `run` subcommand and formats ignore `run_status`.

- [ ] **Step 3: Implement the CLI** (in `cli.py`)

Imports: `from .config import RunEntry` and `from .engine import EntryError, check_paths, collect_files`.

In `_overrides_from`, route `--timeout` by command:

```python
    timeout = getattr(args, "timeout", None) or None
    is_run = getattr(args, "command", None) == "run"
    return config_module.Overrides(
        python=getattr(args, "python", None) or None,
        variadic_rank=getattr(args, "variadic_rank", None) or None,
        ignore=frozenset(getattr(args, "ignore", None) or ()),
        severity=Severity.parse(severity) if severity else None,
        timeout=None if is_run else timeout,
        run_timeout=timeout if is_run else None,
    )
```

In `build_parser`, after `trace`:

```python
    run = sub.add_parser(
        "run", help="run an entry point on meta tensors and report shape bugs, no annotations needed"
    )
    run.add_argument("target", help="file.py::function, or file.py to run its __main__ block")
    run.add_argument("--format", default="full", choices=["full", "concise", "json", "github"])
    _add_common(run)
```

The new command:

```python
def cmd_run(args: argparse.Namespace) -> int:
    entry = RunEntry.parse(args.target, getattr(args, "passthrough", []))
    config = _config_from(args, entry.path)
    try:
        report = check_paths([entry.path], config, run=[entry], trace=False)
    except EntryError as exc:
        print(f"torchtyc run: {exc}", file=sys.stderr)
        return 2
    print(render(report, args.format, config.root))
    if report.worker_error:
        return 2
    return 1 if report.errors else 0
```

In `cmd_check`, pass `run=config.run` to `check_paths` and catch `EntryError` the same way (print, return 2).

In `main`, split off the passthrough before parsing, and register the handler:

```python
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1 :]
    args = build_parser().parse_args(argv)
    args.passthrough = passthrough
    handlers = {
        "check": cmd_check,
        "trace": cmd_trace,
        "run": cmd_run,
        ...
```

Update the module docstring to `"""The command line: check, trace, run, watch, lsp, rules, version."""`.

- [ ] **Step 4: Implement the formats** (in `formats.py`)

```python
_RUN_STATES = {
    "completed": "ran to completion",
    "stopped": "stopped",
    "unsupported": "stopped",
    "error": "stopped",
    "timeout": "timed out",
    "crashed": "crashed",
    "skipped": "not run",
}


def _status_lines(report: Report, root: Path) -> list[str]:
    lines: list[str] = []
    for status in report.run_status:
        text = f"run {_relative_spec(status.entry, root)}: {_RUN_STATES.get(status.state, status.state)}"
        if status.path is not None and status.line is not None:
            text += f" at {_relative(status.path, root)}:{status.line + 1}"
        if status.state == "error":
            text += " (run-error)"
        elif status.state != "completed" and status.reason:
            text += f" ({status.reason})"
        lines.append(text)
        if status.inactive:
            lines.append("  inactive: " + ", ".join(status.inactive))
        if status.ignored:
            lines.append("  ignored: " + ", ".join(status.ignored))
    return lines


def _relative_spec(spec: str, root: Path) -> str:
    path, sep, func = spec.partition("::")
    return _relative(path, root) + (sep + func if sep else "")
```

`_relative_spec` resolves the entry's path against the current directory and shows it relative to the project root. That's why `test_cli_run_exit_codes_and_output`, run from `tools/` with `script.py::main`, expects `run tools/script.py::main`. A path it can't make relative comes back unchanged.

In `_full`, after the diagnostic blocks and before the summary: `if report.run_status: blocks.append("\n".join(_status_lines(report, root)))`. Also render `via` after the source line: `if d.via: body.append(_paint(f"  via: {d.via}", _DIM, color))`.

In `_concise`, insert `lines.extend(_status_lines(report, root))` before the summary.

In `_json`, add `"run_status": [asdict(s) for s in report.run_status],` (and `from dataclasses import asdict`).

In `_github`, after the diagnostics loop:

```python
    for status in report.run_status:
        if status.state != "completed":
            text = _status_lines(Report(run_status=[status]), root)[0]
            lines.append(f"::notice title=torchtyc run::{_escape_workflow(text)}")
```

Also in `_github`, append `%0Avia: ...` when `d.via` is set, just like `note`.

- [ ] **Step 5: Run the tests**

Run: `just test -q`
Expected: PASS (the whole suite).

- [ ] **Step 6: Commit**

```bash
just lint
git add src/torchtyc/cli.py src/torchtyc/formats.py tests/test_run.py tests/test_formats.py
git commit -m "feat: torchtyc run command and run status in every format

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: End-to-end: the motivating case, remote code, a cold cache, blocked network

**Files:**
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: everything above, plus `tests/hf_models.py`.

- [ ] **Step 1: Write the tests** (append to `tests/test_run.py`)

```python
from hf_models import TINY_VOCAB, requires_transformers, write_inputs, write_remote_checkpoint

MOTIVATING = """
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

OUT = Path("/workspace/golden")
CKPTS = ["/workspace/ckpt/a", "/workspace/ckpt/b"]


def load_model(path, dtype, attn_implementation):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    model.eval()
    return model


def load_inputs():
    return load_file(str(OUT / "inputs.safetensors"))


@torch.no_grad()
def run(model, input_ids):
    batch_input = rearrange(input_ids, "seq_len -> 1 seq_len").to("cuda:0")
    logits = model(input_ids=batch_input, use_cache=False)
    rearrange(logits, "1 seq_len -> seq_len ")
    return logits


def main():
    inputs = load_inputs()
    for ckpt in CKPTS:
        model = load_model(ckpt, dtype=torch.float32, attn_implementation="eager")
        for name, input_ids in inputs.items():
            run(model, input_ids)
"""

BUG_TWO = MOTIVATING.replace(
    "logits = model(input_ids=batch_input, use_cache=False)",
    "logits = model(input_ids=batch_input, use_cache=False).logits",
)
FIXED = BUG_TWO.replace('rearrange(logits, "1 seq_len -> seq_len ")', 'rearrange(logits, "1 seq_len vocab -> seq_len vocab")')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path.parent / f"{tmp_path.name}-modules"))  # cold cache
    write_inputs(tmp_path / "golden" / "inputs.safetensors", {"chat": 113, "hindi": 226})
    write_remote_checkpoint(tmp_path / "ckpt" / "a")
    write_remote_checkpoint(tmp_path / "ckpt" / "b")
    return tmp_path


def _line(source: str, needle: str) -> int:
    return next(i for i, text in enumerate(source.splitlines()) if needle in text)


@requires_transformers
@pytest.mark.parametrize(
    ("source", "message", "hint"),
    [
        (MOTIVATING, "Tensor type unknown to einops", "`logits` is a CausalLMOutputWithPast, not a tensor; did you mean `logits.logits`?"),
        (BUG_TWO, "expected 2 dims", "the pattern names 2 axes but the tensor has 3"),
    ],
)
def test_motivating_bugs(project, workspace, source, message, hint):
    path, config = project(source, path_map=(("/workspace", str(workspace)),))
    report = run(path, config)
    (d,) = report.diagnostics
    assert d.rule == "run-error"
    assert d.line == _line(source, 'rearrange(logits, "1 seq_len -> seq_len ")')
    assert d.column == 4
    assert message in d.message
    assert d.hint == hint
    assert "device_map" in report.run_status[0].ignored


@requires_transformers
def test_motivating_fixed_runs_to_completion(project, workspace):
    path, config = project(FIXED, path_map=(("/workspace", str(workspace)),))
    report = run(path, config)
    assert report.diagnostics == []
    assert report.run_status[0].state == "completed"


@requires_transformers
def test_motivating_labels(project, workspace):
    path, config = project(BUG_TWO, path_map=(("/workspace", str(workspace)),))
    (d,) = run(path, config).diagnostics
    assert d.got == f"logits (1, 113 (seq_len?), {TINY_VOCAB} (vocab_size?))"


@requires_transformers
def test_hub_id_without_cache_is_stopped(project):
    source = """
    from transformers import AutoModelForCausalLM

    def main():
        AutoModelForCausalLM.from_pretrained("some-org/not-cached-anywhere")
    """
    path, config = project(source)
    report = run(path, config)
    assert [d.rule for d in report.diagnostics] == ["run-stopped"]
    assert report.diagnostics[0].line == 4


@requires_transformers
def test_module_level_from_pretrained(project, workspace):
    source = """
    from transformers import AutoModelForCausalLM
    import torch

    MODEL = AutoModelForCausalLM.from_pretrained("/workspace/ckpt/a", trust_remote_code=True)

    def main():
        MODEL(input_ids=torch.zeros(1, 5, dtype=torch.long)).logits.view(3)
    """
    path, config = project(source, path_map=(("/workspace", str(workspace)),))
    report = run(path, config)
    assert [d.rule for d in report.diagnostics] == ["run-error"]
    assert report.diagnostics[0].line == 7
```

- [ ] **Step 2: Run them**

Run: `just test tests/test_run.py -q -k "motivating or hub_id or module_level"`
Expected: PASS. A failure here is a real gap in Tasks 4–11. Fix it in the owning module, not in the test, and rerun that task's own tests.

- [ ] **Step 3: Run against the other pins**

```bash
for v in 4.45.2 4.57.6; do uv run --with "transformers==$v" pytest tests/test_run.py -q || echo "FAIL $v"; done
```
Expected: PASS for both. The remote-code fixture uses only `PretrainedConfig`, `PreTrainedModel` and `CausalLMOutputWithPast`. If one major still needs something different, add a fixture for that major in `tests/hf_models.py` (spec §6) rather than changing the test's assertions.

- [ ] **Step 4: Commit**

```bash
git add tests/test_run.py
git commit -m "test: the motivating eval script end to end

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Self-check fixture, CI, README, bench

**Files:**
- Create: `tests/fixtures/run_bugs.py`
- Modify: `pyproject.toml` (`[tool.torchtyc] run`), `.github/workflows/ci.yml`, `justfile` (`self-check`), `README.md`, `bench/bench.py`, `bench/RESULTS.md`

- [ ] **Step 1: Write the fixture**

`tests/fixtures/run_bugs.py`:

```python
"""Wrong on purpose: a shape bug that only `torchtyc run` can see.

Nothing here is annotated, so `torchtyc check` alone reports nothing.
"""

import torch


def project(x):
    w = torch.empty(16, 8)
    return x @ w  # (4, 8) @ (16, 8): the inner sizes differ


def main():
    project(torch.empty(4, 8))


if __name__ == "__main__":
    main()
```

The bug is on line 11 (1-based).

Add to the repo's `pyproject.toml` (create `[tool.torchtyc]` if it doesn't exist; if it does, add only the key):

```toml
[tool.torchtyc]
run = ["tests/fixtures/run_bugs.py::main"]
```

- [ ] **Step 2: Check the self-check finds it**

Run: `just self-check`
Expected: exit 1, with a line starting `::error file=tests/fixtures/run_bugs.py,line=11,` whose title is `torchtyc[run-error]`.

Run: `uv run torchtyc check src`
Expected: the fixture entry is **not** run, because it's outside `src`. The exit code is whatever it was before this plan.

- [ ] **Step 3: Tighten the self-check** (both the `justfile` recipe and the CI step)

After the existing `grep -q "^::error "` check, add:

```bash
if ! grep -q "^::error file=tests/fixtures/run_bugs.py,line=11,.*title=torchtyc\[run-error\]" annotations.txt; then
  echo "torchtyc run found nothing in tests/fixtures/run_bugs.py" >&2
  exit 1
fi
```

In the `justfile`, read the current `self-check` recipe first: it captures output into a variable or file. Add the same grep against that captured output.

- [ ] **Step 4: Add the transformers version job to CI** (`.github/workflows/ci.yml`)

```yaml
  transformers:
    # The meta runtime patches transformers internals that move between
    # releases. Pin the oldest supported 4.x, the last 4.x and the current 5.x.
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        transformers: ["4.45.2", "4.57.6", "5.20.0"]
    env:
      UV_PYTHON: "3.12"
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --all-extras --dev
      - run: uv run --with "transformers==${{ matrix.transformers }}" pytest -q tests/test_metaruntime.py tests/test_run.py
```

Use the pins recorded in Task 1 if the probe changed them.

- [ ] **Step 5: README**

Add a section after "Annotated attributes":

````markdown
## Running unannotated code

Most PyTorch code carries no shape annotations. `torchtyc run` checks it
anyway, by running the script's real entry point with every tensor on the
meta device:

```
torchtyc run tools/eval.py::main            # call main()
torchtyc run tools/eval.py                  # run the `if __name__ == "__main__":` block
torchtyc run tools/eval.py::main -- --n 6   # arguments for argparse
```

No weights load and no kernels run. `from_pretrained` builds the model from
its config on meta, safetensors files are read from their header alone, and
`.to("cuda")` stays on meta, so a multi-billion-parameter model runs a
forward pass in a second on a machine without a GPU. A shape bug raises at
your line:

```
tools/eval.py:36:5: error[run-error]
  `rearrange` raised EinopsError: Wrong shape: expected 2 dims. Received 3-dim tensor.
    Got:      logits (1, 113 (seq_len?), 250624 (vocab_size?))
    36 | rearrange(logits, "1 seq_len -> seq_len ")
  via: main (line 47) → run (line 36)
  hint: the pattern names 2 axes but the tensor has 3
run tools/eval.py::main: stopped at tools/eval.py:36 (run-error)
```

A name in parentheses is a guess: torchtyc prints one only when exactly one
config key or einops axis has that size, and always prints the number.

A run stops, with `run-stopped` at info level, when the script needs a
tensor's value (`.item()`, `if loss < best`, `.tolist()`), a file that is
missing, a download, or a write. That is never reported as an error, and the
status line says how far the run got.

List entry points in `pyproject.toml` and `torchtyc check` runs the ones
under the paths it checks:

```toml
[tool.torchtyc]
run = ["tools/eval.py::main", { entry = "tools/train.py", args = ["--steps", "1"] }]
path-map = { "/workspace" = "." }   # hard-coded container paths
run-timeout = 300                   # seconds per entry
```
````

Add the three rules to the rules table:

```markdown
| `run-error` | error | the entry point raised while running on meta tensors |
| `run-unsupported` | warning | the run reached something the meta runtime cannot stand in for |
| `run-stopped` | info | the run could not continue for a reason that is not a bug |
```

Add the three keys to the configuration section, in the same style as the existing keys.

Append to "Limits":

```markdown
`torchtyc run` is one concrete execution, like the trace pass. Scripts built
on `generate()` usually stop at the `generate` call, because sampling reads
values. A value read made inside C++ (`torch.allclose`) is recognised by its
message, so a bare `except` around it can hide the stop. `path-map` rewrites
paths that loaders and `open` read; it does not rewrite `os.path.exists`,
`os.listdir` or `Path.glob`, so a script that lists a mapped directory needs
the real directory or a symlink. PEFT, DeepSpeed, FSDP and `datasets` are not
supported, and end the run with `run-stopped` or `run-unsupported`.
```

- [ ] **Step 6: Bench**

Add to `bench/bench.py`:

```python
RUN_SCRIPT = '''
import torch
from transformers import LlamaConfig, LlamaForCausalLM

def main():
    config = LlamaConfig(hidden_size=2560, intermediate_size=10240, num_hidden_layers=36,
                         num_attention_heads=32, num_key_value_heads=8, vocab_size=250624,
                         max_position_embeddings=8192)
    model = LlamaForCausalLM(config)
    ids = torch.zeros(1, 4096, dtype=torch.long)
    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
'''


def bench_run(python: str, scratch: Path, repeats: int) -> dict:
    script = scratch / "run_bench.py"
    script.write_text(RUN_SCRIPT)
    from torchtyc.config import Config, RunEntry
    from torchtyc.engine import check_paths

    config = Config(root=scratch, python=python)
    entry = RunEntry(str(script), "main")
    return timed(lambda: check_paths([str(script)], config, run=[entry], trace=False), repeats)
```

In `measure()`, after `worker_startup`:

```python
    results["run_36_layer_4096_tokens"] = bench_run(args.python, scratch, args.repeats)
```

Run: `uv run python bench/bench.py --repeats 3 --out -`
Expected: a `run_36_layer_4096_tokens` entry. Add the number to `bench/RESULTS.md` in the file's existing style.

- [ ] **Step 7: Full CI locally**

Run: `just ci`
Expected: lint passes, every test passes, and the self-check exits 1 with both greps satisfied.

- [ ] **Step 8: Commit**

```bash
git add tests/fixtures/run_bugs.py pyproject.toml .github/workflows/ci.yml justfile README.md bench/bench.py bench/RESULTS.md
git commit -m "docs, ci: torchtyc run self-check, transformers matrix, README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Manual check before release (spec §6, not CI)**

From the `nivida` project:
```bash
uv run --project /home/bhaswata/self_projects/projects/torchtyc torchtyc run src/nivida/tools/parity_hf_gpu.py::main --python .venv/bin/python
```
with `path-map = { "/workspace/ckpt" = "ckpt", "/workspace/golden" = "golden" }` in nivida's `[tool.torchtyc]`. Expected: the same findings the spike reported for the code as written. Report the output to the user; don't commit anything in nivida.
