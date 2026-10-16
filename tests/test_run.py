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
