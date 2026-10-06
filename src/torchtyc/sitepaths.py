"""Which files belong to Python or an installed package, not to the project.

The worker uses this to decide which modules to keep between jobs, and the
trace uses it to tell a project's own `raise` from one inside torch. It imports
nothing heavy, so either side can load it before torch.
"""

from __future__ import annotations

import contextlib
import site
import sys
import sysconfig
from pathlib import Path


def _collect_system_dirs() -> tuple[Path, ...]:
    dirs: set[Path] = set()
    for key in ("stdlib", "platstdlib"):
        val = sysconfig.get_path(key)
        if val:
            dirs.add(Path(val).resolve())
    if hasattr(site, "getsitepackages"):
        with contextlib.suppress(Exception):
            for p in site.getsitepackages():
                dirs.add(Path(p).resolve())
    if getattr(site, "ENABLE_USER_SITE", False) and hasattr(site, "getusersitepackages"):
        with contextlib.suppress(Exception):
            user_site = site.getusersitepackages()
            if isinstance(user_site, str):
                dirs.add(Path(user_site).resolve())
    for p in sys.path:
        if "site-packages" in p or "dist-packages" in p:
            dirs.add(Path(p).resolve())
    return tuple(dirs)


_SYSTEM_DIRS: tuple[Path, ...] = _collect_system_dirs()


def is_system_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
        return any(resolved.is_relative_to(d) for d in _SYSTEM_DIRS)
    except (ValueError, OSError):
        return False
