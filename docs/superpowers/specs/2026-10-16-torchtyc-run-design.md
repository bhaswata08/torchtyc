# `torchtyc run`: shape checking for unannotated code

Status: design, approved section by section on 2026-10-15. Each section had one
adversarial review (Kimi K3, GLM-5.3 and Claude Sonnet), and the accepted
findings are folded in below.

## 1. Goal

> "what can be done about the unannotated code part. A very select number of
> people annotate shapes."

torchtyc today checks only code that carries jaxtyping annotations. Most PyTorch
code, and nearly all Hugging Face research and eval scripts, has none. The
motivating script is an eval loop that loads `inputs.safetensors`, loops over
local checkpoints with
`AutoModelForCausalLM.from_pretrained(path, dtype=..., attn_implementation="eager", trust_remote_code=True, device_map="cuda:0")`,
and calls:

```python
@torch.no_grad()
def run(model, input_ids: torch.Tensor):
    batch_input = rearrange(input_ids, "seq_len -> 1 seq_len").to("cuda:0")
    logits = model(input_ids=batch_input, use_cache=False)   # bug 1: a ModelOutput, not a tensor
    rearrange(logits, "1 seq_len -> seq_len ")                # bug 2: logits is (1, seq, vocab)
    return logits
```

`torchtyc check` reports nothing on it. `torchtyc run` runs the script's real
entry point with every tensor on the meta device and reports both bugs at the
user's line, without loading any weights.

### Success criteria

- Both bugs above are reported as `run-error` at line 36, from three versions
  of the script (as written, with `.logits` added, fully fixed → completes),
  with no annotations.
- A 5B remote-code model builds and runs a full forward pass in seconds on a
  CPU-only machine. The spike measured 0.1–0.3s to build and 1.3s for 2
  checkpoints × 6 inputs, up to 4096 tokens.
- No false errors. Anything the meta device cannot do (read a value, run an op
  without a meta kernel, call a CUDA API with no stub) ends the run with an
  honest info or warning diagnostic, never an error.

### Decisions

1. **Surface:** explicit. `torchtyc run file.py::main` on the CLI, and `check`
   also runs entries listed in config. The LSP comes later, once this is proven.
2. **Data sources in v1:** safetensors, `torch.load`, and real tokenizers on
   CPU. Not `datasets`, numpy or pandas.
3. **Value reads** (`.item()`, `if loss < best`, `.tolist()`) stop the run with
   info-level `run-stopped`. There are no made-up values and no CPU re-runs.
4. **Mechanism:** meta tensors plus targeted patches in one swappable module.
   FakeTensorMode (a private torch API) and a real small CPU run were rejected.

### Out of scope for v1

The LSP and inlay hints of shapes; PEFT/adapters; `accelerate`
`dispatch_model`/`init_empty_weights` (no-ops if called, not patched);
DeepSpeed/FSDP; `datasets`, numpy and pandas as data sources; patching
`generate()`; soundness (a run is one concrete execution, not a proof).

## 2. Architecture

```
cli.py  run file.py[::func] [-- args]  ─┐
cli.py  check   (config `run = [...]`)  ─┤
                                         ▼
engine.py  run_entries(entries, config) -> Report
   · lint pass on the entry files (unchanged)
   · job {"mode": "run", "entries": [...], "path_map": {...}, "run_timeout": ...}
     → run_worker()  (the protocol learns `diagnostic` and `heartbeat` events)
   · lint-scan every project file a diagnostic lands in, apply suppressions,
     make paths relative, render with the existing formats
                                         ▼
worker.py  run_job dispatches on job["mode"]; one forked child per entry,
           or in-process where os.fork is missing (as `check` does today)
                                         ▼
metaruntime.py   (new; imports torch; worker-only, like tracing.py)
   · install() / uninstall(): the whole patch set as one context manager
   · run_entry(path, func, argv) -> streams diagnostics, returns run_status
```

`metaruntime.py` is the only new module that imports torch. The CLI, the
engine and the LSP stay free of torch.

### New rules (`diagnostics.py`, README table)

| Rule | Level | Meaning |
| --- | --- | --- |
| `run-error` | error | the entry point raised, anchored to the deepest frame in a project file |
| `run-stopped` | info | the run could not continue for a reason that is not a bug; the message says which, and that shapes were checked up to that line |
| `run-unsupported` | warning | the run hit something the meta runtime does not support (an op with no meta kernel, an unstubbed CUDA API) |

### New config keys (`config.py`, README)

```toml
[tool.torchtyc]
path-map = { "/workspace" = "." }
run = [{ entry = "src/nivida/tools/parity_hf_gpu.py::main", args = [] }]
run-timeout = 300.0
```

- `path-map`: prefix rewrites applied to every path a patched loader or the
  effects guard's `open` hook reads.
- `run`: the entries `check` also runs, each with an optional argv.
- `run-timeout`: a hard wall-clock limit per entry, in seconds.

## 3. Data flow

### Invocation

```
torchtyc run tools/parity_hf_gpu.py::main            # call main()
torchtyc run tools/parity_hf_gpu.py                  # run the `if __name__ == "__main__":` block
torchtyc run tools/x.py::main -- --ckpt main --n 6   # sets sys.argv for argparse
```

`--format full|concise|json|github` and the common flags (`--python`,
`--allow-effects`, ...) work as they do for `check`. For `run`, `--timeout`
overrides `run-timeout`; the existing `timeout` key keeps its meaning for the
trace pass.

### One entry, step by step

1. The engine runs the lint pass on the entry file and sends a
   `{"mode": "run"}` job to the worker.
2. **Before the call, the engine checks the entry** from the lint pass's AST
   scan, without importing anything. If the file or the
   function doesn't exist, that's a CLI error (exit 2) listing the available
   functions. If the function has required parameters with no default, it gets
   the existing `unresolved-arg` warning and is not run.
3. In the child:
   1. turn on the effects guard;
   2. `metaruntime.install()`, **before** the user module is imported, so
      module-level `from_pretrained` calls and tensor constants also land on
      meta;
   3. set `sys.argv = [entry file, *args]`;
   4. load and call the entry:
      - `::func`: import through the worker's existing `import_from_path`
        under the module's own name (so its `__main__` block does not run),
        then call `func()`.
      - Bare `file.py`: copy `python file.py`, with the script's directory at
        `sys.path[0]` and `runpy.run_path(..., run_name="__main__")`. If the
        file sits inside a package under the project root, copy `python -m`
        with `runpy.run_module(pkg.mod, run_name="__main__")`, so relative
        imports work.
4. While it runs, the child streams each `diagnostic` as it is found, plus a
   `heartbeat` carrying the current user line. **Heartbeats never extend the
   deadline:** `run-timeout` is a hard wall-clock limit. On timeout, the engine
   keeps every diagnostic it has received and adds `run-stopped` ("timed out at
   line N").
5. An exception ends the run and is classified exactly once (§3.3).
6. The engine lint-scans every project file a diagnostic landed in, applies
   suppressions (`# torchtyc: ignore[run-error]` works on the failing line in
   any project file, not only the entry), makes all paths relative, and
   renders the report. Exit codes are the same as `check`: 1 for errors, 2 or
   higher for a failure in torchtyc itself.

### Outcome classification

| Raised | Outcome | Level |
| --- | --- | --- |
| `MetaValueRead` (dispatch mode, Python-level patches, or recognised meta value-read message) | `run-stopped`: needs a value | info |
| `FileNotFoundError`, from any loader, after `path-map` | `run-stopped`: missing input, with the mapped path | info |
| `BlockedEffect` while a patched loader is on the stack (loaders set a context flag) | `run-stopped`: not cached locally, needs network | info |
| `BlockedEffect` anywhere else | the existing blocked-effect diagnostic | as today |
| `Unsupported` (an unstubbed `torch.cuda` call) or `NotImplementedError` from a meta kernel | `run-unsupported` | warning |
| `SystemExit(0)` / `SystemExit(≠0)` | completes / `run-stopped` | — / info |
| Wall-clock timeout | `run-stopped`: timed out at line N; earlier diagnostics kept | info |
| Anything else | `run-error` at the deepest project frame | error |

A *project frame* is a frame whose file is under the project root and not in
site-packages. A bug in your own helper module lands in that module, not at
`main()`.

`MetaValueRead` and `Unsupported` subclass `BaseException`, so a user's
`except Exception:` cannot swallow them. The one exception is the
message-based fallback for value reads made inside C++ (§4.2): that error
starts as a `RuntimeError`, can be swallowed, and is listed as a known limit.

### Status line

Each entry ends with one line, and `--format json` carries the same thing in
a `run_status` field:

```
run tools/parity_hf_gpu.py::main: stopped at line 36 (run-error)
  inactive: transformers is_tracing (not in 4.46.3)
  ignored: device_map, quantization_config
```

A run that stopped early can never look like one that ran to completion.

### Diagnostic format

```
tools/parity_hf_gpu.py:36:5: error[run-error]
  `rearrange` raised: expected 2 dims, received 3
    Got:      logits (1, 113, 250624 (vocab_size?))
    36 | rearrange(logits, "1 seq_len -> seq_len ")
  via: main (line 47) → run (line 36)
  hint: the pattern names 2 axes; `logits` has 3
```

and for the first bug:

```
  `rearrange` raised: `logits` is MoeCausalLMOutputWithPast, not a tensor
  hint: the model returns an output object; did you mean `logits.logits`?
```

**Number labels.** Run mode has a small renderer of its own; it does not
reuse `DimBinder`, so nothing is factored into products. A number gets a
hedged label, `250624 (vocab_size?)`, only when it is at least 128 and appears
exactly once among the keys of the most recently loaded model's config, or is
bound to exactly one einops axis earlier in the same run. The number itself is
always printed, so a wrong label cannot hide the value. The tensors on the
failing line come from the frame's locals, the same way `_shapes_on_line`
finds them.

### `check` and config entries

`check` runs each `run` entry after its normal pass. A `run-error` on the same
line as a `trace-error` from the same file is dropped, so one bug is not
reported twice.

## 4. The meta runtime

torch and transformers come from the project's interpreter. Each patch checks
that its target exists before installing; a missing target makes the patch
inactive, and the status line names it. Section 4.2 is what makes "a skipped
patch only costs coverage" true: any real tensor that slips past a skipped
loader is moved onto meta, not raised as a device mismatch.

### 4.1 Baseline

- `torch.set_default_device("meta")` for the whole run. `torch.zeros`,
  `torch.tensor(...)`, `nn.Linear` parameters and tokenizer output all land on
  meta.
- A **TorchFunctionMode** rewrites every `device=` argument to meta, whatever
  it says. This covers `torch.zeros(n, device="cuda")`,
  `torch.tensor(ids, device=device)` and `x.new_zeros(..., device=...)`, which
  fail before dispatch on a machine without a GPU (verified).

### 4.2 Dispatch mode

A **TorchDispatchMode** (`torch.utils._python_dispatch`):

- **Value reads** raise `MetaValueRead`: `aten::_local_scalar_dense` (which
  covers `.item()`, `bool()` and `x[:t]`), `nonzero` (`where(x)`),
  `masked_select`, the unique ops (`_unique2`, `unique_dim`,
  `unique_consecutive`), `bincount`, `equal`, `repeat_interleave`, and `index`
  with a bool index (boolean masking reaches `nonzero` inside its meta kernel,
  so the mode has to check the index dtype).
- **Device moves:** any non-meta tensor argument is moved onto meta before the
  op runs, so `cpu + meta`, `cat([cpu, meta])`, `from_numpy` results and
  tensors from an unpatched loader all work (verified).
- **`_to_copy` to CPU** stays on meta, so `logits.cpu()` keeps working.
- A **`NotImplementedError` from a meta kernel** becomes `Unsupported`.

Value reads that C++ checks before dispatch need Python-level patches:
`Tensor.tolist`, `Tensor.numpy`, `Tensor.__array__` and `Tensor.__format__`
raise `MetaValueRead` on meta tensors. That covers HF `generate()`'s
`input_ids[:, -1].tolist()` and `f"{loss:.3f}"`. Some C++ paths call `.item()`
internally without dispatching (`torch.allclose`, verified). For those, a
fallback treats a `RuntimeError` or `NotImplementedError` whose message
contains "cannot be called on meta tensors", "unallocated storage" or "Cannot
copy out of meta" as a value read.

### 4.3 Patch table

| Area | Target | Behaviour in a run |
| --- | --- | --- |
| Devices | `Tensor.to/.cuda`, `Module.to/.cuda` | Device → meta; dtype casts still applied. |
| | `Tensor.pin_memory` | Identity on meta. |
| | `torch.cuda` information calls | One fixed fake device: `is_available()` True, `device_count()` 1, `current_device()` 0, capability (9, 0), `is_bf16_supported()` True, `mem_get_info()` 80 GB free, a matching `get_device_properties`/`get_device_name`. `set_device`, `synchronize`, `empty_cache`, `Stream`/`Event`/`current_stream`, `manual_seed*` and RNG state are no-ops. |
| | every other `torch.cuda` callable | Raises `Unsupported` → `run-unsupported`. |
| | `torch.compile` | Returns the function unchanged, so the code runs line by line and anchoring works; listed in the status line. |
| Models | `_BaseAutoModelClass.from_pretrained`, `PreTrainedModel.from_pretrained` | Apply `path-map` → `AutoConfig.from_pretrained(path, trust_remote_code=…, **config overrides)` → `from_config` (Auto classes) or `_from_config` (concrete classes) under meta. Keep `attn_implementation`, and pass `dtype` or `torch_dtype`, whichever that transformers accepts. Ignore `device_map`, `quantization_config`, `low_cpu_mem_usage` and `max_memory`, and list them in the status line. A re-entrancy guard makes a nested `from_pretrained` call (Auto → concrete) run once. |
| | transformers `is_tracing` (5.x) | True for meta tensors. Patched in every `transformers*` module that binds the same object, found by identity through `vars(m)` so lazy modules aren't triggered. |
| | transformers `is_torchdynamo_compiling` | True while the run is active, so HF takes its compile-safe paths, which avoid value reads, on 4.x and 5.x. This may have side effects on dynamo-only code paths, so it is the first thing the implementation verifies against real releases (§6). |
| Data | `safetensors.torch.load_file`, `safe_open` | Apply `path-map`, read only the header, return meta tensors with the real shapes and dtypes. `get_tensor` uses the header the same way. `get_slice` returns a small fake slice with `get_shape`, `get_dtype` and `__getitem__`. |
| | `torch.load` | Apply `path-map`; `map_location="meta"`; `mmap=True` only when the argument is a path, so `BytesIO` still works. |
| | effects guard `open` hook | Apply `path-map` to reads. |
| Tokenizers | none | Run for real from local files; their tensors land on meta through §4.1 and §4.2. |
| Effects | `effects.py` `shutil` copy guards | Check the destination with `_is_allowed_write`, as `open()` does, so `trust_remote_code` can copy modeling files into the modules cache. |
| | `_scratch_dirs` | Add `HF_HOME`, `HF_HUB_CACHE`, `HF_MODULES_CACHE` and `TRANSFORMERS_CACHE`. |

### 4.4 Installing

- `install()` is a context manager that records every patch and restores each
  one by identity on exit. The forked child doesn't strictly need the restore;
  the unit tests do.
- For any target library already in `sys.modules`, the patches go on
  straight away. For the rest, a post-import hook applies them the moment the
  library is imported. Nothing is imported just to patch it.
- The effects-guard changes in §4.3 also apply to `check`. They keep the
  guard's rule of allowing destinations, never callers.

### 4.5 Versions

- torch ≥ 2.1, for a mature meta device and `set_default_device`.
- transformers has no minimum version; patches are detected per feature.
  The README says what each version loses.

### 4.6 Known limits (README Limits section)

- A run is one concrete execution, not a proof.
- Most `generate()`-based scripts end "stopped at line N" at the `generate`
  call, because sampling reads values.
- A value read made inside C++ is caught by message text, so a user's `except`
  can swallow it.
- Unsupported libraries (PEFT, DeepSpeed, FSDP, `datasets`) end the run with
  `run-stopped` or `run-unsupported`, never `run-error`.
- `run` imports the user's code and calls the entry point inside the existing
  effects guard. The README's Security section applies unchanged: the guard
  catches accidents and is not a sandbox.

## 5. Changes by file

| File | Change |
| --- | --- |
| `src/torchtyc/metaruntime.py` | new: §4 |
| `src/torchtyc/worker.py` | `run_job` dispatches on `mode`; `run_entry` in the forked child; anchoring to the deepest project frame; heartbeat and diagnostic events |
| `src/torchtyc/engine.py` | `run_entries`; checking the entry before the call; the protocol learns `diagnostic`/`heartbeat`; hard `run-timeout`; lint-scanning the files diagnostics land in; deduping against `trace-error`; `check` runs config entries |
| `src/torchtyc/cli.py` | the `run` command and `--` passthrough |
| `src/torchtyc/config.py` | `run`, `path-map`, `run-timeout` |
| `src/torchtyc/diagnostics.py` | `run-error`, `run-stopped`, `run-unsupported` |
| `src/torchtyc/effects.py` | `shutil` guards check destinations; HF cache directories |
| `src/torchtyc/formats.py` | the status line; `run_status` in json |
| `README.md` | the command, the rules, the config keys, the limits |
| `pyproject.toml` | dev group: `transformers`, `safetensors`, `tokenizers`; a `run` entry for the self-check fixture |

## 6. Testing

No test touches the network or downloads a model. Every HF model is built from
a `config.json` written into `tmp_path`, and the tokenizer is a tiny WordLevel
`tokenizers` model saved as `tokenizer.json`.

### Hermeticity

An autouse fixture in `tests/conftest.py` sets `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` and a temporary `HF_HOME`, and makes socket `connect`
fail, for every test that touches transformers. Tests that need transformers
are skipped when it isn't installed, so `just test` works in the `nix develop`
shell without it.

### `tests/test_metaruntime.py` (unit, in-process)

`install()` is used only through a fixture that uninstalls in `finally` and
checks, during teardown, that every patched attribute is restored by identity.

- One test per patch-table row: the safetensors header, `safe_open`,
  `get_slice`; `torch.load` with a path and with `BytesIO`; `from_pretrained`
  on an Auto class and on a concrete class, the re-entrancy guard,
  `dtype`/`torch_dtype`; `.to` keeping dtype casts.
- The function mode: factory calls with `device="cuda"`,
  `torch.device("cuda", 0)` and `device=cfg.device`.
- The dispatch mode, from one table: each value read gives `MetaValueRead`
  (`item`, `bool`, masking, `where(x)`, `nonzero`, `repeat_interleave`,
  `equal`, `bincount`, `unique*`, `x[:t]`, `tolist`, `numpy`, `np.array`,
  `__format__`); the fallback messages are recognised (`allclose`); mixed
  CPU/meta arguments are moved; a missing meta kernel gives `Unsupported`.
- The CUDA stubs: the fake device profile, and any other call giving
  `Unsupported`.
- Install ordering: transformers already imported before `install()` gets
  patched straight away; transformers imported after `install()` gets patched
  by the hook.

### `tests/test_run.py` (end-to-end; `project` fixture, real worker)

- **The motivating case:** a tiny Llama-config model and a remote-code
  checkpoint whose `configuration_*.py`/`modeling_*.py` subclass only public,
  stable classes (`PretrainedConfig`, `PreTrainedModel`, a plain `nn.Module`
  body), with the screenshot's `run`/`main`. Three file versions: as in the
  screenshot (`run-error`, bug 1), with `.logits` added (`run-error`, bug 2),
  fully fixed ("ran to completion"). Assert line, column and message text.
- **The HF mask-path canary:** pass an `attention_mask` so the forward takes
  the mask path. With the `is_tracing`/`is_torchdynamo_compiling` patches
  turned off, assert `run-stopped` inside HF. With them on, assert completion.
  This fails if a transformers upgrade makes the patches do nothing.
- **Invocation forms:** `::main`; bare `file.py` with a sibling import; a file
  inside a package (`-m` semantics); argparse with `--` args;
  `SystemExit(0)`/`SystemExit(2)`; an entry that doesn't exist (exit 2);
  required parameters (`unresolved-arg`); module-level `from_pretrained`.
- **Every classification row:** a value read in user code; a value read
  swallowed by `except Exception`; a missing file through a patched loader and
  through plain `open`; `path-map`; a blocked network call inside
  `from_pretrained` against one outside it; a missing meta kernel; an
  unstubbed CUDA call; and a timeout with `run-timeout = 2` and a `time.sleep`
  loop after a finding, which keeps the earlier finding.
- A cold `HF_MODULES_CACHE` in `tmp_path` with `trust_remote_code`.
- Anchoring in a helper module, with `# torchtyc: ignore[run-error]` there;
  relative paths; `--format github` and `json` (`run_status`); `check` running
  config entries; deduping against `trace-error`.
- Number labels: config labels only for the latest model, values below 128
  left bare, no factoring.

### `tests/test_effects.py`

The `shutil` copy guards allow listed destinations and block others; the
`HF_*` cache variables are honoured.

### CI

- The existing 3.11/3.12/3.13 test job runs with `transformers`, `safetensors`
  and `tokenizers` from the dev group, pinned by `uv.lock`.
- A new transformers version job on Python 3.12 runs `test_metaruntime.py`
  and `test_run.py` against three pinned releases: the oldest 4.x minor we
  support, the last 4.x, and the current 5.x. The exact versions are chosen in
  the implementation plan. Anything specific to one major gets a fixture of
  its own for that major.
- The self-check job also greps for the exact `run-error` annotation on
  `tests/fixtures/run_bugs.py` at its line, so a `run` that silently does
  nothing fails CI.

### Self-check fixture

`tests/fixtures/run_bugs.py` is a plain-torch script (no HF) with deliberate
run bugs, listed as a `run` entry in the repo's `[tool.torchtyc]`.
`just self-check` covers `run` and still expects exit 1.

### Bench

`bench/bench.py` times a run of a 36-layer, 4096-token config forward on
meta, recorded in `bench/RESULTS.md`.

### Manual check before release

Run `torchtyc run` on the motivating script (`parity_hf_gpu.py`, in the user's
`nivida` project) and compare with the spike's results.

## 7. Open risks

1. **`is_torchdynamo_compiling → True` side effects.** HF's compile paths may
   call dynamo-only APIs (`torch._dynamo.mark_static`, `torch.compiler.*`).
   Verified first in the implementation, against the three pinned releases.
   If it misbehaves, fall back to patching only `is_tracing` and accept 4.x
   coverage loss (status line: inactive).
2. **Patch drift across transformers releases.** Mitigated by feature
   detection, the canary test and the version job; not eliminated.
3. **Value reads inside C++** are caught by message text, which can change
   between torch releases. The unit test table pins the current messages.
