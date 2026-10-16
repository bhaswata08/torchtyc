"""A malformed pyproject.toml must never take the language server down."""

import textwrap
from pathlib import Path

from torchtyc import config as config_module
from torchtyc.diagnostics import Severity


def write(tmp_path, table: str):
    (tmp_path / "pyproject.toml").write_text(textwrap.dedent(table))
    return tmp_path


def test_good_values_are_read(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        severity = "warning"
        variadic-rank = 3
        timeout = 12.5
        ignore = ["unused-dim"]
        """,
    )
    config = config_module.load(root)
    assert config.severity is Severity.WARNING
    assert config.variadic_rank == 3
    assert config.timeout == 12.5
    assert config.ignore == frozenset({"unused-dim"})


def test_unknown_severity_falls_back_to_the_default(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        severity = "warn"
        """,
    )
    assert config_module.load(root).severity is Severity.INFO


def test_non_numeric_values_fall_back_to_the_default(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        variadic-rank = "two"
        timeout = "soon"
        """,
    )
    config = config_module.load(root)
    assert config.variadic_rank == 2
    assert config.timeout == 60.0


def test_out_of_range_values_fall_back_to_the_default(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        variadic-rank = 0
        timeout = 0
        """,
    )
    config = config_module.load(root)
    assert config.variadic_rank == 2
    assert config.timeout == 60.0


def test_wrongly_typed_lists_fall_back_to_the_default(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        ignore = "unused-dim"
        exclude = 3
        """,
    )
    config = config_module.load(root)
    assert config.ignore == frozenset()
    assert config.exclude == (".venv", "build", "dist", "__pycache__", ".git")


def test_find_root_multiple_paths_common_ancestor(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    p_a = a / "model.py"
    p_b = b / "model.py"
    p_a.touch()
    p_b.touch()

    root = config_module.find_root([p_a, p_b])
    assert root == tmp_path


def test_run_keys_are_read(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        run = ["tools/a.py::main", { entry = "tools/b.py", args = ["--n", "2"] }]
        path-map = { "/workspace" = ".", "/workspace/golden" = "data/golden" }
        run-timeout = 12
        """,
    )
    config = config_module.load(root)
    assert [e.spec for e in config.run] == [
        str(root / "tools/a.py") + "::main",
        str(root / "tools/b.py"),
    ]
    assert config.run[1].args == ("--n", "2")
    # Longest prefix first, so the more specific mapping wins.
    assert config.path_map == (
        ("/workspace/golden", str((root / "data/golden").resolve())),
        ("/workspace", str(root.resolve())),
    )
    assert config.run_timeout == 12.0


def test_bad_run_keys_keep_defaults(tmp_path):
    root = write(
        tmp_path,
        """
        [tool.torchtyc]
        run = "not-a-list"
        path-map = ["nope"]
        run-timeout = -1
        """,
    )
    config = config_module.load(root)
    assert config.run == ()
    assert config.path_map == ()
    assert config.run_timeout == 300.0


def test_run_entry_parse():
    entry = config_module.RunEntry.parse("x.py::main", ["--a"])
    assert (entry.path, entry.func, entry.args, entry.spec) == (
        "x.py",
        "main",
        ("--a",),
        "x.py::main",
    )
    bare = config_module.RunEntry.parse("x.py")
    assert (bare.func, bare.spec) == (None, "x.py")


def test_run_timeout_override():
    config = config_module.Config(root=Path("."))
    config_module.Overrides(run_timeout=3.0).apply(config)
    assert config.run_timeout == 3.0
    assert config.timeout == 60.0
