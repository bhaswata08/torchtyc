"""A malformed pyproject.toml must never take the language server down."""

import textwrap

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
