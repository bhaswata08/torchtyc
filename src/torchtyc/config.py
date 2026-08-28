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


@dataclass(frozen=True)
class Overrides:
    """Options given on a command line, which win over any pyproject.toml.

    They are kept apart from Config so a long-running server can still discover
    the project a file belongs to and then apply what the user asked for on top.
    """

    python: str | None = None
    variadic_rank: int | None = None
    ignore: frozenset[str] = frozenset()
    severity: Severity | None = None
    timeout: float | None = None

    def apply(self, config: Config) -> Config:
        if self.python is not None:
            config.python = self.python
        if self.variadic_rank is not None:
            config.variadic_rank = self.variadic_rank
        if self.ignore:
            config.ignore = config.ignore | self.ignore
        if self.severity is not None:
            config.severity = self.severity
        if self.timeout is not None:
            config.timeout = self.timeout
        return config


def find_root(start: Path) -> Path:
    start = start.resolve()
    for directory in [start, *start.parents]:
        if (directory / "pyproject.toml").exists():
            return directory
    return start if start.is_dir() else start.parent


def _strings(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a list, got {type(value).__name__}")
    return [str(item) for item in value]


def _at_least(value: int, minimum: int) -> int:
    if value < minimum:
        raise ValueError(f"{value} is below {minimum}")
    return value


def _positive(value: float) -> float:
    if value <= 0:
        raise ValueError(f"{value} is not positive")
    return value


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

    # A bad value keeps its default rather than raising. The language server
    # builds its config while starting, and a typo in pyproject.toml must not
    # take the whole server down with a traceback the editor cannot show.
    def read(key: str, attribute: str, convert) -> None:
        if key not in table:
            return
        try:
            value = convert(table[key])
        except (TypeError, ValueError):
            return
        setattr(config, attribute, value)

    read("severity", "severity", lambda v: Severity.parse(str(v)))
    read("ignore", "ignore", lambda v: frozenset(_strings(v)))
    read("exclude", "exclude", lambda v: tuple(_strings(v)))
    read("variadic-rank", "variadic_rank", lambda v: _at_least(int(v), 1))
    read("python", "python", str)
    read("einops", "einops", bool)
    read("timeout", "timeout", lambda v: _positive(float(v)))
    read("extra-paths", "extra_paths", lambda v: tuple(_strings(v)))

    return config
