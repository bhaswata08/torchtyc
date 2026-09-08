"""Tests for side-effect containment during import and tracing."""

from __future__ import annotations

import signal
import socket
import sys
import textwrap
from pathlib import Path

import pytest

from torchtyc.config import Config
from torchtyc.effects import BlockedEffect, active_guard, unwrap_blocked
from torchtyc.engine import check_paths


@pytest.fixture
def project(tmp_path: Path):
    def build(source: str, **config_kwargs) -> tuple[list[str], Config]:
        path = tmp_path / "model.py"
        path.write_text(textwrap.dedent(source))
        config = Config(root=tmp_path, python=sys.executable, **config_kwargs)
        return [str(path)], config

    return build


HEADER = """
    import torch
    from jaxtyping import Float
    from torch import Tensor
"""


def test_pathlib_write_text_blocked(project, tmp_path: Path):
    target = tmp_path / "side_effect.txt"
    paths, config = project(
        HEADER
        + f"""
    from pathlib import Path

    Path({str(target)!r}).write_text("should_not_exist")

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "BlockedEffect" in diag.message
    assert "cannot write" in diag.message
    # Confirm it anchored to the write line
    lines = Path(paths[0]).read_text().splitlines()
    assert "write_text" in lines[diag.line]


def test_builtin_open_write_blocked(project, tmp_path: Path):
    target = tmp_path / "opened.txt"
    paths, config = project(
        HEADER
        + f"""
    with open({str(target)!r}, "w") as f:
        f.write("data")

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "cannot open" in diag.message
    lines = Path(paths[0]).read_text().splitlines()
    assert "open(" in lines[diag.line]


def test_builtin_open_append_blocked(project, tmp_path: Path):
    target = tmp_path / "appended.txt"
    paths, config = project(
        HEADER
        + f"""
    with open({str(target)!r}, "a") as f:
        f.write("data")

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    assert "cannot open" in report.diagnostics[0].message


def test_os_makedirs_blocked(project, tmp_path: Path):
    target = tmp_path / "created_dir"
    paths, config = project(
        HEADER
        + f"""
    import os
    os.makedirs({str(target)!r})

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "cannot make directory" in diag.message
    lines = Path(paths[0]).read_text().splitlines()
    assert "makedirs" in lines[diag.line]


def test_os_remove_blocked(project, tmp_path: Path):
    target = tmp_path / "existing.txt"
    target.write_text("keep_me")
    paths, config = project(
        HEADER
        + f"""
    import os
    os.remove({str(target)!r})

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "cannot remove" in diag.message


def test_shutil_rmtree_blocked(project, tmp_path: Path):
    dir_target = tmp_path / "dir_to_remove"
    dir_target.mkdir()
    paths, config = project(
        HEADER
        + f"""
    import shutil
    shutil.rmtree({str(dir_target)!r})

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert dir_target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "cannot remove directory tree" in diag.message


def test_pathlib_touch_blocked(project, tmp_path: Path):
    target = tmp_path / "touched.txt"
    paths, config = project(
        HEADER
        + f"""
    from pathlib import Path
    Path({str(target)!r}).touch()

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "cannot touch" in diag.message


def test_pathlib_mkdir_blocked(project, tmp_path: Path):
    target = tmp_path / "new_dir"
    paths, config = project(
        HEADER
        + f"""
    from pathlib import Path
    Path({str(target)!r}).mkdir()

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "cannot make directory" in diag.message


def test_network_connect_blocked(project):
    paths, config = project(
        HEADER
        + """
    import socket
    s = socket.socket()
    s.connect(("1.1.1.1", 80))

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "outbound network connection" in diag.message
    lines = Path(paths[0]).read_text().splitlines()
    assert "s.connect" in lines[diag.line]


def test_urllib_network_blocked(project):
    paths, config = project(
        HEADER
        + """
    import urllib.request
    urllib.request.urlopen("http://example.com", timeout=1)

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "BlockedEffect" in diag.message
    assert "outbound network connection" in diag.message
    lines = Path(paths[0]).read_text().splitlines()
    assert "urlopen" in lines[diag.line]


def test_reads_at_import_time_allowed(project, tmp_path: Path):
    config_file = tmp_path / "cfg.txt"
    config_file.write_text("hello_config")
    paths, config = project(
        HEADER
        + f"""
    from pathlib import Path
    data = Path({str(config_file)!r}).read_text()

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.ok


def test_escape_hatch_allows_effects(project, tmp_path: Path):
    target = tmp_path / "allowed_side_effect.txt"
    paths, config = project(
        HEADER
        + f"""
    from pathlib import Path
    Path({str(target)!r}).write_text("permitted")

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """,
        allow_effects=True,
    )
    report = check_paths(paths, config)
    assert target.exists()
    assert target.read_text() == "permitted"
    assert report.diagnostics == []
    assert report.ok


def test_side_effect_in_imported_helper(tmp_path: Path):
    helper = tmp_path / "helper.py"
    side_effect_file = tmp_path / "helper_effect.txt"
    helper.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            Path({str(side_effect_file)!r}).write_text("from_helper")
            """
        )
    )

    main = tmp_path / "main.py"
    main.write_text(
        textwrap.dedent(
            """
            import torch
            from jaxtyping import Float
            from torch import Tensor
            import helper

            def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x
            """
        )
    )

    config = Config(
        root=tmp_path,
        python=sys.executable,
        extra_paths=(str(tmp_path),),
    )
    report = check_paths([str(main)], config)
    assert not side_effect_file.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "import-error"
    assert "helper.py" in diag.message
    assert "cannot write" in diag.message


def test_direct_active_guard_context():
    with active_guard(enabled=True), pytest.raises(BlockedEffect):
        socket.socket().connect(("127.0.0.1", 9999))
    s = socket.socket()
    assert s.connect != active_guard


def test_constructor_write_text_blocked(project, tmp_path: Path):
    target = tmp_path / "side_effect_init.txt"
    paths, config = project(
        HEADER
        + f"""
    from torch import nn
    from pathlib import Path

    class Block(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            Path({str(target)!r}).write_text("constructed")
            self.W = nn.Parameter(torch.empty((d_model, d_model)))

        def forward(self, x: Float[Tensor, "b d_model"]) -> Float[Tensor, "b d_model"]:
            return x @ self.W
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "trace-error"
    assert "BlockedEffect" in diag.message
    assert "cannot write" in diag.message
    lines = Path(paths[0]).read_text().splitlines()
    assert "write_text" in lines[diag.line]


def test_constructor_write_allowed_by_config(project, tmp_path: Path):
    target = tmp_path / "allowed_init_effect.txt"
    paths, config = project(
        HEADER
        + f"""
    from torch import nn
    from pathlib import Path

    class Block(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            Path({str(target)!r}).write_text("constructed")
            self.W = nn.Parameter(torch.empty((d_model, d_model)))

        def forward(self, x: Float[Tensor, "b d_model"]) -> Float[Tensor, "b d_model"]:
            return x @ self.W
    """,
        allow_effects=True,
    )
    report = check_paths(paths, config)
    assert target.exists()
    assert target.read_text() == "constructed"
    assert report.diagnostics == []
    assert report.ok


def test_trace_forward_write_blocked(project, tmp_path: Path):
    target = tmp_path / "forward_effect.txt"
    paths, config = project(
        HEADER
        + f"""
    from pathlib import Path

    def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        Path({str(target)!r}).write_text("forward_executed")
        return x
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "trace-error"
    assert "BlockedEffect" in diag.message
    assert "cannot write" in diag.message
    lines = Path(paths[0]).read_text().splitlines()
    assert "write_text" in lines[diag.line]


def test_attribute_check_write_blocked(project, tmp_path: Path):
    target = tmp_path / "attr_effect.txt"
    paths, config = project(
        HEADER
        + f"""
    from torch import nn
    from pathlib import Path

    class Model(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            Path({str(target)!r}).write_text("in_init")
            self.weights: Float[nn.Parameter, "d d"] = nn.Parameter(torch.empty((d, d)))
    """
    )
    report = check_paths(paths, config)
    assert not target.exists()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "trace-error"
    assert "BlockedEffect" in diag.message
    lines = Path(paths[0]).read_text().splitlines()
    assert "write_text" in lines[diag.line]


def test_unwrap_blocked_breaks_cyclic_cause():
    first = ValueError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first

    def timeout_handler(signum, frame):
        raise TimeoutError("unwrap_blocked did not return within timeout")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(2)
    try:
        assert unwrap_blocked(first) is None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_unwrap_blocked_finds_blocked_in_cyclic_cause():
    blocked = BlockedEffect("blocked effect")
    first = ValueError("first")
    first.__cause__ = blocked
    blocked.__cause__ = first

    assert unwrap_blocked(first) is blocked
