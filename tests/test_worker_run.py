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
    diagnostics, status, _ = run(
        tmp_path, "import torch\ndef main():\n    torch.empty(2, 3) @ torch.empty(3, 4)\n"
    )
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
    source = (
        "import torch\nimport sib\nif __name__ == '__main__':\n    torch.empty(sib.VALUE).view(2)\n"
    )
    diagnostics, _, _ = run(tmp_path, source, func=None, files=files)
    assert [d["rule"] for d in diagnostics] == ["run-error"]


def test_package_file_runs_like_dash_m(tmp_path):
    files = {"pkg/__init__.py": "", "pkg/util.py": "N = 5\n"}
    (tmp_path / "pkg").mkdir()
    script = tmp_path / "pkg" / "tool.py"
    script.write_text(
        "import torch\nfrom .util import N\nif __name__ == '__main__':\n    torch.empty(N).view(2)\n"
    )
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


def test_keyerror_entry_result(tmp_path):
    source = 'def main():\n    raise KeyError("entry_result")\n'
    diagnostics, status, events = run(tmp_path, source)
    assert len(diagnostics) == 1
    assert [e["event"] for e in events].count("entry_result") == 1
    assert status["state"] == "error"


def test_no_heartbeat_after_entry_result(tmp_path):
    source = "import time\ndef main():\n    time.sleep(2.5)\n"
    _, status, events = run(tmp_path, source)
    assert status["state"] == "completed"
    entry_result_idx = next(i for i, e in enumerate(events) if e.get("event") == "entry_result")
    heartbeats = [i for i, e in enumerate(events) if e.get("event") == "heartbeat"]
    assert heartbeats
    assert all(i < entry_result_idx for i in heartbeats)


def test_import_from_path_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod_rel.py").write_text("VALUE = 123\n")
    mod = worker.import_from_path(Path("mod_rel.py"))
    assert mod.VALUE == 123
    assert Path(mod.__file__).is_absolute()


def test_torch_save_blocked_does_not_overwrite(tmp_path):  # Review Focus 1
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt_file = ckpt_dir / "best.pt"
    original_bytes = b"precious_checkpoint_data_do_not_overwrite"
    ckpt_file.write_bytes(original_bytes)

    source = """
    import torch
    def main():
        torch.save({"w": torch.empty(1000, 1000)}, "ckpt/best.pt")
    """
    diagnostics, status, _ = run(tmp_path, source)
    assert [d["rule"] for d in diagnostics] == ["run-stopped"]
    assert status["state"] == "stopped"
    assert ckpt_file.read_bytes() == original_bytes


def test_safetensors_save_file_blocked(tmp_path):
    source = """
    import torch
    from safetensors.torch import save_file
    def main():
        save_file({"w": torch.empty(10, 10)}, "weights.safetensors")
    """
    diagnostics, status, _ = run(tmp_path, source)
    assert [d["rule"] for d in diagnostics] == ["run-stopped"]
    assert status["state"] == "stopped"
    assert not (tmp_path / "weights.safetensors").exists()


def test_safetensors_save_gives_run_stopped(tmp_path):
    source = """
    import torch
    from safetensors.torch import save
    def main():
        save({"w": torch.empty(10, 10)})
    """
    diagnostics, status, _ = run(tmp_path, source)
    assert [d["rule"] for d in diagnostics] == ["run-stopped"]
    assert status["state"] == "stopped"


def test_stdout_fd1_redirected_in_child(tmp_path):
    source = """
    import os, torch
    def main():
        os.write(1, b'{"event": "entry_result", "status": {"entry": "x", "state": "completed"}}\\n')
        torch.empty(2, 3) @ torch.empty(4, 5)
    """
    diagnostics, status, events = run(tmp_path, source)
    assert [d["rule"] for d in diagnostics] == ["run-error"]
    assert status["state"] == "error"
    entry_results = [e for e in events if e.get("event") == "entry_result"]
    assert len(entry_results) == 1
