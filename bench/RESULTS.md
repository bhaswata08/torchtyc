# Benchmark results

Every number below comes from one run of `python bench/bench.py`, on NixOS with
torch 2.12 (CPU, 16 threads), Python 3.13, on a 24-core machine. Numbers are
medians over the harness's repeats. Re-run the harness rather than trusting
these; they exist to record the shape of the costs, not to be precise. The
absolute figures move with the machine, but the ratios do not.

## The three paths, separately

| Path | Cost | Runs when |
| --- | --- | --- |
| Lint (AST only) | 0.35 ms per module | every keystroke |
| Trace (subprocess) | ~1.2 s | on open, on save, after 0.7 s idle |
| Marginal trace per function | ~1 ms | within a trace |

## Lint scales with the file, and stays cheap

From `scaling_by_module_count[*].lint`:

| Annotated modules in one file | Lint |
| --- | --- |
| 1 | 0.35 ms |
| 8 | 2.29 ms |
| 32 | 9.14 ms |

This is the path that runs while you type, and it never imports torch, so the
project's dependencies cannot slow it down.

## Tracing is dominated by importing torch, not by tracing

From `worker_startup` and `scaling_by_module_count[*].check`:

| Step | Cost |
| --- | --- |
| Spawn a bare Python | 15 ms |
| `import torch` | 862 ms |
| `import torch, einops, jaxtyping` | 879 ms |
| Worker process with an empty job | 883 ms |
| Full `check` of a 1-module file | 1180 ms |
| Full `check` of a 32-module file | 1207 ms |

Going from 1 annotated module to 32 costs **27 ms**, about 1 ms per function.
Roughly 75% of a check is `import torch`, which happens once per check
regardless of how much there is to check. einops and jaxtyping together add
17 ms on top of torch, and the process itself costs 15 ms.

## Meta tensors versus real ones

One `einsum` of a `(256, width)` against a `(width, width)`, from
`forward_pass_by_width`:

| width | real CPU | meta | speedup |
| --- | --- | --- | --- |
| 512 | 0.33 ms | 0.026 ms | 13x |
| 2048 | 3.66 ms | 0.026 ms | 140x |
| 8192 | 177.07 ms | 0.026 ms | 6686x |

Meta is flat, because no arithmetic happens and no memory is allocated. Real
CPU grows with the work. The gap therefore widens with model size, which is the
whole reason the checker traces on meta: checking a large model costs the same
as checking a small one.

## Measuring a real project

`bench.py --target <path>` lints and checks every `.py` file under a directory
and reports both under `targets`. That is how the numbers for a real codebase
are produced; they are not included here because they depend on the project.

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
