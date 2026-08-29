"""Benchmark torchtyc.

Measures the three costs that matter separately, because they land on very
different paths:

  lint    the AST pass, which runs on every keystroke in the editor
  trace   the subprocess pass, which imports torch and runs the model on meta
  real    the same forward pass on real CPU tensors, for comparison

The last one is the point of the whole design: if tracing on meta were not
dramatically cheaper than running the model, there would be no reason to do it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


def timed(fn, repeats: int) -> dict[str, float]:
    """Time `fn` `repeats` times, after one warm-up call that is not recorded.

    The first call to any of these paths pays a one-off cost the later ones do
    not: a thread pool spinning up, an import landing in a cache, a page being
    faulted in. Recording it would put a number in the table that no second run
    reproduces.
    """
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
        "n": repeats,
    }


def make_model(path: Path, layers: int) -> None:
    """A file with `layers` annotated modules, to measure scaling."""
    head = textwrap.dedent("""
        import torch
        from einops import einsum
        from jaxtyping import Float
        from torch import Tensor, nn
    """)
    body = "".join(
        textwrap.dedent(f"""

        class Block{i}(nn.Module):
            def __init__(self, d_in: int, d_out: int) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(
                    torch.empty((d_out, d_in))
                )

            def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
                return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        """)
        for i in range(layers)
    )
    path.write_text(head + body)


def bench_lint(paths: list[str], repeats: int) -> dict:
    from torchtyc.config import Config
    from torchtyc.discovery import scan_source
    from torchtyc.engine import lint_scan

    config = Config(root=Path("."))
    sources = {p: Path(p).read_text() for p in paths}

    def run():
        for path, text in sources.items():
            lint_scan(scan_source(text, path), config)

    return timed(run, repeats)


def bench_trace(paths: list[str], python: str, repeats: int) -> dict:
    from torchtyc.config import Config
    from torchtyc.engine import check_paths

    config = Config(root=Path("."), python=python)

    def run():
        report = check_paths(paths, config)
        # A worker that never started is not a fast check, it is a broken one.
        if report.worker_error is not None:
            raise RuntimeError(f"the worker failed: {report.worker_error}")

    return timed(run, repeats)


def bench_spawn(python: str, body: str, repeats: int) -> dict:
    """Spawn a fresh interpreter that runs `body` and exits."""

    def run():
        # An interpreter that cannot import torch exits fast, and that speed
        # would be published as a measurement. Fail instead.
        subprocess.run(
            [python, "-c", body],
            capture_output=True,
            text=True,
            check=True,
        )

    return timed(run, repeats)


def bench_worker_startup(python: str, repeats: int) -> dict:
    """The fixed cost of a check, broken into the steps that make it up.

    A check pays for a process, then for `import torch`, then for the rest of
    the imports. Measuring them separately is what says where the ~1.3 s of a
    check actually goes, and therefore what is worth attacking.
    """
    job = json.dumps({"paths": [], "variadic_rank": 2, "sources": {}, "hover": False})
    package_root = str(Path(__file__).resolve().parent.parent / "src")

    def run_worker():
        subprocess.run(
            [
                python,
                "-c",
                (
                    "import sys; sys.path.append(sys.argv[1]); "
                    "import torch; "
                    "from torchtyc.worker import main; raise SystemExit(main())"
                ),
                package_root,
            ],
            input=job,
            capture_output=True,
            text=True,
            check=True,
        )

    return {
        "bare_python": bench_spawn(python, "pass", repeats),
        "import_torch": bench_spawn(python, "import torch", repeats),
        "import_torch_einops_jaxtyping": bench_spawn(
            python, "import torch, einops, jaxtyping", repeats
        ),
        "empty_worker": timed(run_worker, repeats),
    }


def bench_real_vs_meta(width: int, batch: int, repeats: int) -> dict:
    """The same forward pass, on real CPU tensors and on meta tensors.

    The operands are built once per device and outside the timed region. A real
    `(width, width)` allocation is 256 MB at width 8192 and a meta one is free,
    so timing the build would charge the real device for memory the comparison
    is not about and inflate the ratio.
    """
    import torch
    from einops import einsum

    def build(device: str):
        # `torch.ones`, not `torch.empty`: uninitialised memory can hold
        # denormal floats, and denormal arithmetic runs an order of magnitude
        # slower on x86. That cost belongs to the garbage in the buffer, not to
        # the einsum, and it lands unpredictably depending on which pages the
        # allocator hands back.
        w = torch.ones((width, width), device=device)
        x = torch.ones((batch, width), device=device)
        return x, w

    real_x, real_w = build("cpu")
    meta_x, meta_w = build("meta")

    def run(x, w):
        einsum(x, w, "b d_in, d_out d_in -> b d_out")

    return {
        "real_cpu": timed(lambda: run(real_x, real_w), repeats),
        "meta": timed(lambda: run(meta_x, meta_w), repeats),
    }


def bench_width_sweep(widths: tuple[int, ...], batch: int, repeats: int) -> dict:
    """Real versus meta across model widths.

    Meta is flat in the tensor size and real CPU is not, so the ratio only means
    something as a curve. One width would hide that.
    """
    return {width: bench_real_vs_meta(width, batch, repeats) for width in widths}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="-")
    parser.add_argument("--target", action="append", default=[])
    args = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="torchtyc-bench-"))

    results: dict = {"python": args.python}

    # Scaling with the number of annotated modules in one file.
    scaling = {}
    for layers in (1, 8, 32):
        path = scratch / f"model_{layers}.py"
        make_model(path, layers)
        scaling[layers] = {
            "lint": bench_lint([str(path)], args.repeats * 4),
            "check": bench_trace([str(path)], args.python, args.repeats),
        }
    results["scaling_by_module_count"] = scaling

    results["worker_startup"] = bench_worker_startup(args.python, args.repeats)
    results["forward_pass_by_width"] = bench_width_sweep((512, 2048, 8192), 256, args.repeats * 4)

    for target in args.target:
        files = [str(p) for p in Path(target).rglob("*.py") if ".venv" not in str(p)]
        results.setdefault("targets", {})[target] = {
            "files": len(files),
            "lint": bench_lint(files, args.repeats * 2),
            "check": bench_trace(files, args.python, args.repeats),
        }

    text = json.dumps(results, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
