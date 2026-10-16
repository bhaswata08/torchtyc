"""Configuration, read from `[tool.torchtyc]` in the nearest pyproject.toml."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostics import Severity


@dataclass(frozen=True)
class RunEntry:
    """One entry point for `torchtyc run`: a file, an optional function, its argv."""

    path: str
    func: str | None = None
    args: tuple[str, ...] = ()

    @property
    def spec(self) -> str:
        return f"{self.path}::{self.func}" if self.func else self.path

    @classmethod
    def parse(cls, text: str, args: Sequence[str] = ()) -> RunEntry:
        path, _, func = text.partition("::")
        return cls(path=path, func=func or None, args=tuple(args))


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
    allow_effects: bool = False
    # `torchtyc run`: entries `check` also runs, prefix rewrites for
    # hard-coded paths, and the wall-clock limit for one entry.
    run: tuple[RunEntry, ...] = ()
    path_map: tuple[tuple[str, str], ...] = ()
    run_timeout: float = 300.0

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
    allow_effects: bool | None = None
    run_timeout: float | None = None

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
        if self.allow_effects is not None:
            config.allow_effects = self.allow_effects
        if self.run_timeout is not None:
            config.run_timeout = self.run_timeout
        return config


def find_root(start: Path | str | Sequence[Path | str]) -> Path:
    if isinstance(start, (list, tuple)):
        if not start:
            return find_root(Path("."))
        if len(start) == 1:
            return find_root(start[0])
        resolved = [Path(p).resolve() for p in start]
        try:
            common = Path(os.path.commonpath(resolved))
        except ValueError:
            return find_root(resolved[0])
        common_dir = common if common.is_dir() else common.parent
        for directory in [common_dir, *common_dir.parents]:
            if (directory / "pyproject.toml").exists():
                return directory
        return common_dir

    start = Path(start).resolve()
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


def _run_entries(value: object, root: Path) -> tuple[RunEntry, ...]:
    if not isinstance(value, list):
        raise TypeError("run must be a list")
    entries: list[RunEntry] = []
    for item in value:
        if isinstance(item, str):
            text, args = item, ()
        elif isinstance(item, dict) and isinstance(item.get("entry"), str):
            text, args = item["entry"], tuple(_strings(item.get("args", [])))
        else:
            raise TypeError("each run entry is a string or a table with `entry`")
        entry = RunEntry.parse(text, args)
        path = Path(entry.path)
        entries.append(
            RunEntry(str(path if path.is_absolute() else root / path), entry.func, entry.args)
        )
    return tuple(entries)


def _path_map(value: object, root: Path) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise TypeError("path-map must be a table")
    pairs = []
    for old, new in value.items():
        target = Path(str(new)).expanduser()
        pairs.append(
            (
                str(old).rstrip("/") or "/",
                str((target if target.is_absolute() else root / target).resolve()),
            )
        )
    return tuple(sorted(pairs, key=lambda pair: -len(pair[0])))


def load(start: Path | str | Sequence[Path | str] = ".") -> Config:
    root = find_root(start)
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
    read("allow-effects", "allow_effects", bool)
    read("run", "run", lambda v: _run_entries(v, root))
    read("path-map", "path_map", lambda v: _path_map(v, root))
    read("run-timeout", "run_timeout", lambda v: _positive(float(v)))

    return config
