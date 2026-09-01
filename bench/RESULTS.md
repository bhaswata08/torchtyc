# Benchmark results

Every number below comes from one run of `python bench/bench.py --repeats 30`,
on Linux with torch 2.13 (CPU, 16 threads), Python 3.14, on a 24-core machine.
Numbers are medians over the harness's repeats, after a warm-up call the harness
discards. Re-run the harness rather than trusting these; they exist to record
the shape of the costs, not to be precise. The absolute figures move with the
machine, but the ratios do not.

## The three paths, separately

| Path | Cost | Runs when |
| --- | --- | --- |
| Lint (AST only) | 0.27 ms per module | every keystroke |
| Trace (subprocess) | ~1.3 s | on open, on save, after 0.7 s idle |
| Marginal trace per function | ~1 ms | within a trace |

## Lint scales with the file, and stays cheap

From `scaling_by_module_count[*].lint`:

| Annotated modules in one file | Lint |
| --- | --- |
| 1 | 0.27 ms |
| 8 | 1.76 ms |
| 32 | 7.15 ms |

This is the path that runs while you type, and it never imports torch, so the
project's dependencies cannot slow it down.

## Tracing is dominated by importing torch, not by tracing

From `worker_startup` and `scaling_by_module_count[*].check`:

| Step | Cost |
| --- | --- |
| Spawn a bare Python | 16 ms |
| `import torch` | 993 ms |
| `import torch, einops, jaxtyping` | 1002 ms |
| Worker process with an empty job | 1007 ms |
| Full `check` of a 1-module file | 1266 ms |
| Full `check` of a 32-module file | 1298 ms |

Going from 1 annotated module to 32 costs **32 ms**, about 1 ms per function.
Roughly 78% of a check is `import torch`, which happens once per check
regardless of how much there is to check. einops and jaxtyping together add
9 ms on top of torch, and the process itself costs 16 ms.

## Meta tensors versus real ones

One `einsum` of a `(256, width)` against a `(width, width)`, from
`forward_pass_by_width`. Both operands are built before the timer starts, so the
row measures the einsum and nothing else:

| width | real CPU | meta | speedup |
| --- | --- | --- | --- |
| 512 | 0.25 ms | 0.017 ms | 15x |
| 2048 | 3.54 ms | 0.022 ms | 164x |
| 8192 | 58.52 ms | 0.018 ms | 3305x |

Meta is flat, because no arithmetic happens. Real CPU grows with the work. The
gap therefore widens with model size, which is the whole reason the checker
traces on meta: checking a large model costs the same as checking a small one.

Meta also allocates nothing, but that saving does not show up here, because the
operands are built outside the timed region on both sides. It shows up instead
as memory the checker never needs: the width-8192 real operands are 256 MB,
against zero on meta.

## Measuring a real project

`bench.py --target <path>` lints and checks every `.py` file under a directory
and reports both under `targets`. That is how the numbers for a real codebase
are produced; they are not included here because they depend on the project.

## Where the remaining time goes

The fixed ~1 s is a cold `import torch` in a fresh subprocess. The design
deliberately spawns a fresh process per check so that an edit on disk is what
gets checked and no module state goes stale. That correctness is worth the
latency for a CLI run, and the editor hides most of it behind a debounce, but
it is the obvious thing to attack if the 1.3 s ever feels slow:

- **Keep a warm worker.** Hold one subprocess open with torch already imported
  and drop only the user's modules from `sys.modules` between requests. That
  would cut an editor trace from ~1.3 s to ~300 ms. The cost is that user
  module side effects, and any C-extension state they touch, accumulate in a
  long-lived process, which is exactly what the fresh-process design avoids.
- **Skip the trace when nothing relevant changed.** Hash the annotated
  signatures and the file's imports; if neither moved, republish the previous
  trace diagnostics instead of re-running. Cheap, safe, and removes most traces
  during ordinary editing.

Neither is implemented. The lint pass already covers the keystroke path, so
the trace latency only shows up as a delay before shape errors appear.
