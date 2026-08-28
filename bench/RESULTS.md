# Benchmark results

Measured on NixOS, torch 2.12 (CPU), Python 3.13, via `python bench/bench.py`.
Numbers are medians. Re-run the harness rather than trusting these; they exist
to record the shape of the costs, not to be precise.

## The three paths, separately

| Path | Cost | Runs when |
| --- | --- | --- |
| Lint (AST only) | 0.34 ms per module | every keystroke |
| Trace (subprocess) | ~1.2 s | on open, on save, after 0.7 s idle |
| Marginal trace per function | ~1 ms | within a trace |

## Lint scales with the file, and stays cheap

| Annotated modules in one file | Lint |
| --- | --- |
| 1 | 0.34 ms |
| 8 | 2.28 ms |
| 32 | 9.00 ms |

The whole `cs336_basics` package (6 files) lints in **5.9 ms**. This is the path
that runs while you type, and it never imports torch, so it cannot be slowed
down by the project's dependencies.

## Tracing is dominated by importing torch, not by tracing

| Step | Cost |
| --- | --- |
| Spawn a bare Python | 18 ms |
| `import torch` | 880 ms |
| `import torch, einops, jaxtyping` | 889 ms |
| Full `check` of a 1-module file | 1187 ms |
| Full `check` of a 32-module file | 1217 ms |

Going from 1 annotated module to 32 costs **30 ms**, about 1 ms per function.
Roughly 75% of a check is `import torch`, which happens once per check
regardless of how much there is to check.

`torchtyc check cs336_basics` (6 files, 3 annotated functions): **1666 ms**.

## Meta tensors versus real ones

One `einsum` of a `(256, width)` against a `(width, width)`:

| width | real CPU | meta | speedup |
| --- | --- | --- | --- |
| 512 | 0.39 ms | 0.027 ms | 15x |
| 2048 | 3.67 ms | 0.026 ms | 142x |
| 8192 | 57.32 ms | 0.026 ms | 2206x |

Meta is flat, because no arithmetic happens and no memory is allocated. Real
CPU grows with the work. The gap therefore widens with model size, which is the
whole reason the checker traces on meta: checking a large model costs the same
as checking a small one.

## Where the remaining time goes

The fixed ~900 ms is a cold `import torch` in a fresh subprocess. The design
deliberately spawns a fresh process per check so that an edit on disk is what
gets checked and no module state goes stale. That correctness is worth the
latency for a CLI run, and the editor hides most of it behind a debounce, but
it is the obvious thing to attack if the 1.2 s ever feels slow:

- **Keep a warm worker.** Hold one subprocess open with torch already imported
  and drop only the user's modules from `sys.modules` between requests. That
  would cut an editor trace from ~1.2 s to ~300 ms. The cost is that user
  module side effects, and any C-extension state they touch, accumulate in a
  long-lived process, which is exactly what the fresh-process design avoids.
- **Skip the trace when nothing relevant changed.** Hash the annotated
  signatures and the file's imports; if neither moved, republish the previous
  trace diagnostics instead of re-running. Cheap, safe, and removes most traces
  during ordinary editing.

Neither is implemented. The lint pass already covers the keystroke path, so
the trace latency only shows up as a delay before shape errors appear.
