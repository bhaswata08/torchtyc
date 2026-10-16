"""`torchtyc run` end to end: a real project in tmp_path, a real worker."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import pytest
from hf_models import TINY_VOCAB, requires_transformers, write_inputs, write_remote_checkpoint

from torchtyc.config import Config, RunEntry
from torchtyc.engine import EntryError, check_paths, entry_problem


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
    second, _ = project(
        "import time\n\ndef main():\n    while True:\n        time.sleep(0.05)\n",
        name="second.py",
    )
    entries = [RunEntry(str(first), "main"), RunEntry(str(second), "main")]
    report = check_paths([str(first), str(second)], config, run=entries, trace=False)
    rules = {(Path(d.path).name, d.rule) for d in report.diagnostics}
    assert rules == {("first.py", "run-error"), ("second.py", "run-stopped")}
    (stopped,) = [d for d in report.diagnostics if d.rule == "run-stopped"]
    assert "timed out after 2s" in stopped.message
    assert stopped.line in (3, 4)  # inside the loop, from the last heartbeat
    assert [s.state for s in report.run_status] == ["error", "timeout"]


def test_timeout_restarts_worker_for_later_entries(project):
    slow, config = project(
        "import time\n\ndef main():\n    while True:\n        time.sleep(0.05)\n",
        name="slow.py",
        run_timeout=2.0,
    )
    bug, _ = project(BUG, name="bug.py")
    entries = [RunEntry(str(slow), "main"), RunEntry(str(bug), "main")]
    report = check_paths([str(slow), str(bug)], config, run=entries, trace=False)
    rules = {(Path(d.path).name, d.rule) for d in report.diagnostics}
    assert rules == {("slow.py", "run-stopped"), ("bug.py", "run-error")}
    (stopped,) = [d for d in report.diagnostics if d.rule == "run-stopped"]
    assert "timed out after 2s" in stopped.message
    assert [s.state for s in report.run_status] == ["timeout", "error"]
    assert report.errors == 1


def test_sigterm_handling_child_times_out_cleanly(project):
    source = """
    import signal
    import time

    def main():
        signal.signal(signal.SIGTERM, lambda *a: None)
        while True:
            time.sleep(0.05)
    """
    path, config = project(source, name="sigterm_trap.py", run_timeout=2.0)
    start = time.monotonic()
    report = check_paths([str(path)], config, run=[RunEntry(str(path), "main")], trace=False)
    elapsed = time.monotonic() - start
    assert elapsed < 15.0
    assert [s.state for s in report.run_status] == ["timeout"]
    assert any(d.rule == "run-stopped" and "timed out" in d.message for d in report.diagnostics)


def test_suppression_in_helper_file(project, tmp_path):
    (tmp_path / "helpers.py").write_text(
        "import torch\ndef bad(x):\n    return x.view(7)  # torchtyc: ignore[run-error]\n"
    )
    path, config = project(
        "import torch\nfrom helpers import bad\ndef main():\n    bad(torch.empty(3))\n"
    )
    report = run(path, config)
    assert report.diagnostics == []


def test_suppression_in_entry_file_is_used(project):
    path, config = project(
        "import torch\ndef main():\n    torch.empty(3).view(7)  # torchtyc: ignore[run-error]\n"
    )
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


from torchtyc.cli import main as cli_main


def test_cli_run_exit_codes_and_output(project, capsys, monkeypatch, tmp_path):
    project(BUG, name="tools/script.py")
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
    project(BUG, name="tools/script.py")
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.torchtyc]\npython = "{sys.executable}"\nrun = ["tools/script.py::main"]\n'
    )
    assert cli_main(["check", str(tmp_path / "tools"), "--format", "github"]) == 1
    out = capsys.readouterr().out
    assert "title=torchtyc[run-error]" in out
    # A file outside the entry's directory: the config entry is not run, so a
    # clean check stays clean (exit 0, no run-error).
    (tmp_path / "elsewhere.py").write_text("x = 1\n")
    assert cli_main(["check", str(tmp_path / "elsewhere.py")]) == 0


def test_entry_file_with_invalid_utf8_raises_entry_error(tmp_path, project):
    path = tmp_path / "bad_bytes.py"
    path.write_bytes(b"\xff\xfe def main(): pass")
    entry = RunEntry(str(path), "main")
    with pytest.raises(EntryError, match="cannot be read"):
        entry_problem(entry)
    _, config = project("def main(): pass")
    with pytest.raises(EntryError, match="cannot be read"):
        run(path, config)


def test_worker_killed_mid_run_marks_uncompleted_entries_crashed(project):
    first, config = project("def main():\n    pass\n", name="first.py")
    second, _ = project(
        "import os, signal\ndef main():\n    os.kill(os.getppid(), signal.SIGKILL)\n",
        name="second.py",
    )
    entries = [RunEntry(str(first), "main"), RunEntry(str(second), "main")]
    report = check_paths([str(first), str(second)], config, run=entries, trace=False)
    assert report.worker_error is not None
    assert "exit code" in report.worker_error
    assert len(report.run_status) == 2
    assert report.run_status[0].state == "completed"
    assert report.run_status[1].state == "crashed"
    assert report.run_status[1].reason == "the worker stopped before this entry finished"
    assert not report.ok


def test_timeout_before_any_entry_started(project, monkeypatch):
    first, config = project("def main():\n    pass\n")
    entries = [RunEntry(str(first), "main")]
    monkeypatch.setattr(
        "torchtyc.engine._communicate",
        lambda *args, **kwargs: ([], None, True),
    )
    report = check_paths([str(first)], config, run=entries, trace=False)
    assert (
        report.worker_error
        == f"the run timed out after {config.run_timeout:g}s before any entry started"
    )
    assert len(report.run_status) == 1
    assert report.run_status[0].state == "crashed"
    assert report.run_status[0].reason == "the worker stopped before this entry finished"
    assert not report.ok


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
FIXED = BUG_TWO.replace(
    'rearrange(logits, "1 seq_len -> seq_len ")',
    'rearrange(logits, "1 seq_len vocab -> seq_len vocab")',
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "HF_MODULES_CACHE", str(tmp_path.parent / f"{tmp_path.name}-modules")
    )  # cold cache
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
        (
            MOTIVATING,
            "Tensor type unknown to einops",
            "`logits` is a CausalLMOutputWithPast, not a tensor; did you mean `logits.logits`?",
        ),
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
