# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

torchtyc checks PyTorch tensor shapes against jaxtyping annotations. It builds each
annotated function's arguments on `torch.device("meta")`, calls the function, and compares
the traced shapes with the annotations. `README.md` is the user-facing spec: the rules
table, the config keys, the exit codes and the diagnostic formats all live there, so keep
it in step with behaviour changes.

## Commands

Dependencies go through uv (`uv add`, never `uv pip install`). The `justfile` wraps
everything; `just` lists the recipes.

```bash
just sync                         # uv sync --all-extras --dev
just test                         # uv run pytest; extra args pass through
just test tests/test_binding.py -k prime    # one file / one test
just lint                         # ruff check + ruff format --check over src tests bench
just fmt                          # ruff format + ruff check --fix
just self-check                   # torchtyc check tests/fixtures --format github
just ci                           # lint, test, self-check: what CI runs
uv run python bench/bench.py      # lint / trace / real-CPU timings (see bench/RESULTS.md)
```

CI runs the test job on Python 3.11, 3.12 and 3.13. The fixtures in `tests/fixtures/`
are wrong on purpose. Ruff excludes them, and `self-check` expects exit 1 (findings).
Exit 2 or higher means torchtyc itself failed.

The version lives only in `src/torchtyc/__init__.py` (`__version__`); hatch reads it from
there. Use `just bump X.Y.Z`, `just release-dry X.Y.Z`, or `just release X.Y.Z`. The last
one publishes to PyPI with the token in `.env` and pushes a tag, so run it only when asked.

## Architecture

Checking has two passes with different costs. `engine.py` joins them.

1. **Lint pass (in-process, no imports of user code, no torch).** `discovery.py` walks the
   AST to find annotated targets, how to construct their owning class, annotated `self.`
   attributes, einops calls and `# torchtyc: ignore` comments. `annotations.py` parses
   jaxtyping dim strings from AST nodes without evaluating them. `einops_rules.py` checks
   einops patterns statically. This pass must stay cheap and must never fail, because the
   LSP runs it on every change, including on files that do not import.

2. **Trace pass (subprocess, project interpreter).** `engine.py` starts
   `python -c <bootstrap>` under the *project's* interpreter (found from `.venv`/`venv`
   beside the nearest `pyproject.toml`, or `--python`/config). The bootstrap *appends*
   torchtyc's source root to `sys.path` so the project's own torch and jaxtyping win.
   Do not change this to `PYTHONPATH`. `worker.py` reads one JSON job on stdin, imports
   the user module and writes one JSON result on stdout. `tracing.py` is the only module
   that imports torch, and only the worker loads it (lazily). Keep torch imports out of
   every other module, because the CLI and LSP may run under an interpreter without torch.

Supporting modules:

- `binding.py`: `DimBinder` gives each dimension name a distinct prime, starting at 101,
  and matches traced shapes to specs. Products of primes are factored back into names
  (`d_model*seq`) for messages. A trace that fails because primes do not divide
  (`d_model // n_heads`) is retried on widths taken from literals in the model, which
  gives `trace-retried`.
- `effects.py`: a guard, active while the worker imports and traces, that blocks
  filesystem writes, network (including DNS) and process spawning. It catches accidents
  and is not a security boundary. `allow-effects` turns it off.
- `diagnostics.py`: the `RULES` table. Rule names are public API (CLI output, ignore
  comments, editor codes), so renaming one is a breaking change. Add rules here and to the
  README table.
- `config.py`: `[tool.torchtyc]` from the nearest `pyproject.toml`. CLI flags override it.
- `formats.py`: `full`, `concise`, `json` and `github` renderers for a `Report`.
- `cli.py`: the `check`, `trace`, `watch`, `lsp`, `rules` and `version` commands.
- `lsp.py`: a pygls server. It publishes lint results on every change and traces only on
  open, on save and after 0.7s of quiet. Stale worker runs are cancelled through
  `engine.terminate_worker` (process-group SIGTERM, then SIGKILL).

Optional dependencies (`pygls`, `watchfiles`, `einops`) are extras, so import them only
where they are needed.

## Tests

`tests/test_engine.py` and `tests/test_lsp.py` are end-to-end: they write a real project
into `tmp_path` (the `project` fixture), then run the real subprocess and the real torch
import. The other test files cover one module each. Diagnostics anchor to the user's line
(not to torch internals or a shared layer further down), so tests assert on
line/column and on message text.

## Dev shell

`nix develop` (`flake.nix`) gives a nixpkgs Python 3.13 with torch and the dev tools, and
sets `PYTHONPATH=$PWD/src`. It is an alternative to the uv venv on NixOS hosts without
nix-ld.
