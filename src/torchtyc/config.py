"""Configuration, read from `[tool.torchtyc]` in the nearest pyproject.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostics import Severity


@dataclass
class Config:
    root: Path
    # Diagnostics below this severity are dropped before anything is printed.
    severity: Severity = Severity.INFO
    ignore: frozenset[str] = frozenset()
    exclude: tuple[str, ...] = (".venv", "build", "dist", "__pycache__", ".git")
    variadic_rank: int = 2
    # The interpreter that imports project code. Defaults to the venv beside
    # pyproject.toml, because that is the one with the project's torch.
    python: str | None = None
    einops: bool = True
    timeout: float = 60.0
    extra_paths: tuple[str, ...] = field(default_factory=tuple)

    @property
    def interpreter(self) -> str:
        if self.python:
            return str(Path(self.python).expanduser())
        for candidate in (
            self.root / ".venv" / "bin" / "python",
            self.root / "venv" / "bin" / "python",
        ):
            if candidate.exists():
                return str(candidate)
        import sys

        return sys.executable


def find_root(start: Path) -> Path:
    start = start.resolve()
    for directory in [start, *start.parents]:
        if (directory / "pyproject.toml").exists():
            return directory
    return start if start.is_dir() else start.parent


def load(start: Path | str = ".") -> Config:
    root = find_root(Path(start))
    config = Config(root=root)

    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return config

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return config

    table = data.get("tool", {}).get("torchtyc", {})
    if not isinstance(table, dict):
        return config

    if "severity" in table:
        config.severity = Severity.parse(str(table["severity"]))
    if "ignore" in table:
        config.ignore = frozenset(str(r) for r in table["ignore"])
    if "exclude" in table:
        config.exclude = tuple(str(p) for p in table["exclude"])
    if "variadic-rank" in table:
        config.variadic_rank = int(table["variadic-rank"])
    if "python" in table:
        config.python = str(table["python"])
    if "einops" in table:
        config.einops = bool(table["einops"])
    if "timeout" in table:
        config.timeout = float(table["timeout"])
    if "extra-paths" in table:
        config.extra_paths = tuple(str(p) for p in table["extra-paths"])

    return config
