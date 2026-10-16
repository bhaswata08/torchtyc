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
