"""Containment for side effects in the code being checked.

Checking runs user code twice over: importing a module executes its top-level
statements, and tracing constructs classes and calls methods. This module
installs a guard around both that blocks filesystem writes and outbound network
calls while letting reads through.

Python cannot enforce an in-process security boundary. Code that deliberately
aims to circumvent this guard can do so. The guard exists to catch accidental
effects such as automated downloads, hardware initialization, and logging.
"""

from __future__ import annotations

import builtins
import getpass
import inspect
import io
import os
import pathlib
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Ensure tempfile has initialized its default candidate directory before
# any hooks run. Python probes candidate temp directories with write tests.
tempfile.gettempdir()


class BlockedEffect(PermissionError):
    """Raised when user code attempts a forbidden side effect during import."""

    def __init__(
        self,
        message: str,
        target: str | None = None,
        culprit: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.target = target
        self.culprit = culprit


def unwrap_blocked(exc: BaseException) -> BlockedEffect | None:
    """Find a BlockedEffect inside an exception's cause or context chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, BlockedEffect):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


_contexts: list[str] = []


def _current_phase() -> str:
    return _contexts[-1] if _contexts else "import"


def _caller_info() -> tuple[str, int] | None:
    """Find the first frame outside this module."""
    here = Path(__file__).resolve()
    frame = inspect.currentframe()
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename and not filename.startswith("<"):
            try:
                resolved = Path(filename).resolve()
                if resolved != here:
                    return filename, frame.f_lineno
            except OSError:
                pass
        frame = frame.f_back
    return None


def _is_write_mode(mode: str) -> bool:
    return any(char in mode for char in ("w", "a", "x", "+"))


def _allowed_cache_dirs() -> list[Path]:
    dirs: list[Path] = []

    # TorchInductor cache
    inductor_env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if inductor_env:
        try:
            dirs.append(Path(inductor_env).resolve())
        except (ValueError, OSError):
            pass
    try:
        username = getpass.getuser()
    except (KeyError, OSError):
        getuid = getattr(os, "getuid", None)
        username = f"uid_{getuid()}" if callable(getuid) else "unknown_user"
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", username)
    default_name = f"torchinductor_{sanitized_username}"
    dirs.append((Path(tempfile.gettempdir()) / default_name).resolve())
    dirs.append((Path("/var/tmp") / default_name).resolve())

    # Triton caches
    for env_var in ("TRITON_CACHE_DIR", "TRITON_DUMP_DIR", "TRITON_OVERRIDE_DIR"):
        val = os.environ.get(env_var)
        if val:
            try:
                dirs.append(Path(val).resolve())
            except (ValueError, OSError):
                pass

    triton_home = os.environ.get("TRITON_HOME")
    if triton_home:
        try:
            dirs.append((Path(triton_home) / ".triton").resolve())
        except (ValueError, OSError):
            pass
    else:
        try:
            dirs.append((Path.home() / ".triton").resolve())
        except (RuntimeError, OSError):
            pass

    return dirs


def _scratch_dirs() -> list[Path]:
    """Destinations that are scratch space, not exfiltration or persistence.

    The system temp directory (via `tempfile.gettempdir()`, which already
    respects `TMPDIR`, plus an explicit read of `TMPDIR`/`TEMP`/`TMP` in case
    the environment changed after `tempfile` cached its answer) and the user
    cache directory (`XDG_CACHE_HOME`, falling back to `~/.cache`). Libraries
    such as matplotlib (font cache), joblib, numba, and HuggingFace hubs write
    here during import; blocking that pushes users to `allow-effects = true`,
    which switches the guard off wholesale and protects less, not more.

    Only destinations are allowed, never callers: anything writing outside
    these directories, the inductor/triton caches, is still blocked no matter
    which module it comes from.
    """
    dirs: list[Path] = []
    try:
        dirs.append(Path(tempfile.gettempdir()).resolve())
    except (ValueError, OSError):
        pass
    for env_var in ("TMPDIR", "TEMP", "TMP"):
        val = os.environ.get(env_var)
        if val:
            try:
                dirs.append(Path(val).resolve())
            except (ValueError, OSError):
                pass
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    try:
        if xdg_cache:
            dirs.append(Path(xdg_cache).resolve())
        else:
            dirs.append((Path.home() / ".cache").resolve())
    except (RuntimeError, ValueError, OSError):
        pass
    return dirs


def attempt_cooperative_stop(thread: threading.Thread) -> None:
    """Attempt cooperative shutdown of a thread the checked code spawned.

    Only cooperative hooks such as `cancel` or `stop` are safe to call.
    Asynchronous exception injection can interrupt a thread holding internal
    locks or blocked in C calls, which risks deadlocking or corrupting the
    process. Unstoppable threads are left alone for the caller to report.
    """
    if not thread.is_alive():
        return
    if hasattr(thread, "cancel") and callable(thread.cancel):
        try:
            thread.cancel()
        except Exception:  # noqa: BLE001, S110
            pass
    if hasattr(thread, "stop") and callable(thread.stop):
        try:
            thread.stop()
        except Exception:  # noqa: BLE001, S110
            pass
    thread.join(timeout=0.05)


def _is_allowed_write(target: Any) -> bool:
    # Importing torch and tracing on meta tensors populates the inductor and
    # triton caches. Those writes come from us, not from the user's code, and
    # blocking them would break the check we are trying to run. Scratch files
    # under the system temp directory and the user cache directory are likewise
    # allowed: a font or dataset cache landing in /tmp or ~/.cache is not a
    # dataset download, telemetry, or checkpoint write. The checked project
    # tree stays blocked even when it sits under one of those directories
    # (a pytest `tmp_path` project lives under /tmp, as does any project
    # someone keeps directly in /tmp): a checkpoint or source overwrite inside
    # the project is exactly what the guard is for.
    try:
        if isinstance(target, bytes):
            target = os.fsdecode(target)
        p = Path(target).resolve()
        for allowed in _allowed_cache_dirs():
            try:
                if p.is_relative_to(allowed):
                    return True
            except (TypeError, ValueError):
                continue
        for allowed in _scratch_dirs():
            try:
                if p.is_relative_to(allowed) and not _is_protected(p):
                    return True
            except (TypeError, ValueError):
                continue
    except (TypeError, ValueError, OSError):
        return False
    return False


_protected_stack: list[tuple[Path, ...]] = []


def _is_protected(p: Path) -> bool:
    """Whether a resolved path sits inside a tree the current check protects."""
    for roots in _protected_stack:
        for root in roots:
            try:
                if p.is_relative_to(root):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _resolve_roots(roots: Any) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for root in roots or ():
        try:
            resolved.append(Path(root).resolve())
        except (TypeError, ValueError, OSError):
            continue
    return tuple(resolved)


_orig_builtins_open = builtins.open
_orig_io_open = io.open

_orig_os_open = os.open
_orig_os_remove = os.remove
_orig_os_unlink = os.unlink
_orig_os_rmdir = os.rmdir
_orig_os_removedirs = os.removedirs
_orig_os_mkdir = os.mkdir
_orig_os_makedirs = os.makedirs
_orig_os_rename = os.rename
_orig_os_renames = os.renames
_orig_os_replace = os.replace
_orig_os_link = os.link
_orig_os_symlink = os.symlink
_orig_os_chmod = os.chmod
_orig_os_chown = os.chown
_orig_os_lchmod = getattr(os, "lchmod", None)
_orig_os_lchown = getattr(os, "lchown", None)
_orig_os_truncate = os.truncate
_orig_os_ftruncate = os.ftruncate
_orig_os_utime = os.utime

_orig_shutil_copy = shutil.copy
_orig_shutil_copy2 = shutil.copy2
_orig_shutil_copyfile = shutil.copyfile
_orig_shutil_copytree = shutil.copytree
_orig_shutil_move = shutil.move
_orig_shutil_rmtree = shutil.rmtree
_orig_shutil_chown = shutil.chown

_orig_path_touch = pathlib.Path.touch
_orig_path_mkdir = pathlib.Path.mkdir
_orig_path_rmdir = pathlib.Path.rmdir
_orig_path_unlink = pathlib.Path.unlink
_orig_path_rename = pathlib.Path.rename
_orig_path_replace = pathlib.Path.replace
_orig_path_symlink_to = pathlib.Path.symlink_to
_orig_path_hardlink_to = pathlib.Path.hardlink_to
_orig_path_chmod = pathlib.Path.chmod
_orig_path_lchmod = getattr(pathlib.Path, "lchmod", None)
_orig_path_write_text = pathlib.Path.write_text
_orig_path_write_bytes = pathlib.Path.write_bytes
_orig_path_open = pathlib.Path.open

_orig_socket_connect = socket.socket.connect
_orig_socket_connect_ex = socket.socket.connect_ex
_orig_socket_sendto = socket.socket.sendto
_orig_create_connection = socket.create_connection
_orig_socket_getaddrinfo = socket.getaddrinfo
_orig_socket_gethostbyname = socket.gethostbyname
_orig_socket_gethostbyname_ex = socket.gethostbyname_ex
_orig_socket_gethostbyaddr = socket.gethostbyaddr
_orig_socket_getnameinfo = socket.getnameinfo

_orig_subprocess_popen = subprocess.Popen
_orig_subprocess_popen_init = subprocess.Popen.__init__
_orig_subprocess_run = subprocess.run
_orig_subprocess_call = subprocess.call
_orig_subprocess_check_call = subprocess.check_call
_orig_subprocess_check_output = subprocess.check_output
_orig_subprocess_getoutput = getattr(subprocess, "getoutput", None)
_orig_subprocess_getstatusoutput = getattr(subprocess, "getstatusoutput", None)

_orig_os_system = os.system
_orig_os_posix_spawn = getattr(os, "posix_spawn", None)
_orig_os_posix_spawnp = getattr(os, "posix_spawnp", None)
_orig_os_spawnl = getattr(os, "spawnl", None)
_orig_os_spawnle = getattr(os, "spawnle", None)
_orig_os_spawnlp = getattr(os, "spawnlp", None)
_orig_os_spawnlpe = getattr(os, "spawnlpe", None)
_orig_os_spawnv = getattr(os, "spawnv", None)
_orig_os_spawnve = getattr(os, "spawnve", None)
_orig_os_spawnvp = getattr(os, "spawnvp", None)
_orig_os_spawnvpe = getattr(os, "spawnvpe", None)
_orig_os_execl = getattr(os, "execl", None)
_orig_os_execle = getattr(os, "execle", None)
_orig_os_execlp = getattr(os, "execlp", None)
_orig_os_execlpe = getattr(os, "execlpe", None)
_orig_os_execv = getattr(os, "execv", None)
_orig_os_execve = getattr(os, "execve", None)
_orig_os_execvp = getattr(os, "execvp", None)
_orig_os_execvpe = getattr(os, "execvpe", None)


def _guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
    mode = args[0] if args else kwargs.get("mode", "r")
    if isinstance(mode, str) and _is_write_mode(mode) and not _is_allowed_write(file):
        culprit = _caller_info()
        raise BlockedEffect(
            f"cannot open '{file}' for writing during {_current_phase()}",
            target=str(file),
            culprit=culprit,
        )
    return _orig_builtins_open(file, *args, **kwargs)


def _guarded_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
    write_flags = (
        os.O_WRONLY
        | os.O_RDWR
        | getattr(os, "O_CREAT", 0)
        | getattr(os, "O_TRUNC", 0)
        | getattr(os, "O_APPEND", 0)
    )
    if (flags & write_flags) and not _is_allowed_write(path):
        culprit = _caller_info()
        raise BlockedEffect(
            f"cannot open '{path}' for writing during {_current_phase()}",
            target=str(path),
            culprit=culprit,
        )
    return _orig_os_open(path, flags, *args, **kwargs)


def _guarded_os_remove(path: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(path):
        return _orig_os_remove(path, *args, **kwargs)
    raise BlockedEffect(
        f"cannot remove '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_os_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(path):
        return _orig_os_unlink(path, *args, **kwargs)
    raise BlockedEffect(
        f"cannot unlink '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_os_rmdir(path: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot remove directory '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_os_removedirs(name: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot remove directories '{name}' during {_current_phase()}",
        target=str(name),
        culprit=_caller_info(),
    )


def _guarded_os_mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(path):
        return _orig_os_mkdir(path, *args, **kwargs)
    raise BlockedEffect(
        f"cannot make directory '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_os_makedirs(name: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(name):
        return _orig_os_makedirs(name, *args, **kwargs)
    raise BlockedEffect(
        f"cannot make directory '{name}' during {_current_phase()}",
        target=str(name),
        culprit=_caller_info(),
    )


def _guarded_os_rename(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(src) and _is_allowed_write(dst):
        return _orig_os_rename(src, dst, *args, **kwargs)
    raise BlockedEffect(
        f"cannot rename '{src}' during {_current_phase()}", target=str(src), culprit=_caller_info()
    )


def _guarded_os_renames(old: Any, new: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(old) and _is_allowed_write(new):
        return _orig_os_renames(old, new, *args, **kwargs)
    raise BlockedEffect(
        f"cannot rename '{old}' during {_current_phase()}", target=str(old), culprit=_caller_info()
    )


def _guarded_os_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(src) and _is_allowed_write(dst):
        return _orig_os_replace(src, dst, *args, **kwargs)
    raise BlockedEffect(
        f"cannot replace '{src}' during {_current_phase()}", target=str(src), culprit=_caller_info()
    )


def _guarded_os_link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot link '{src}' during {_current_phase()}", target=str(src), culprit=_caller_info()
    )


def _guarded_os_symlink(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot symlink '{src}' during {_current_phase()}", target=str(src), culprit=_caller_info()
    )


def _guarded_os_chmod(path: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot chmod '{path}' during {_current_phase()}", target=str(path), culprit=_caller_info()
    )


def _guarded_os_chown(path: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot chown '{path}' during {_current_phase()}", target=str(path), culprit=_caller_info()
    )


def _guarded_os_truncate(path: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot truncate '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_os_ftruncate(fd: int, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot truncate file descriptor during {_current_phase()}", culprit=_caller_info()
    )


def _guarded_os_utime(path: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot update timestamp for '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_shutil_copy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
    raise BlockedEffect(
        f"cannot copy to '{dst}' during {_current_phase()}", target=str(dst), culprit=_caller_info()
    )


def _guarded_shutil_copy2(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
    raise BlockedEffect(
        f"cannot copy to '{dst}' during {_current_phase()}", target=str(dst), culprit=_caller_info()
    )


def _guarded_shutil_copyfile(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
    raise BlockedEffect(
        f"cannot copy to '{dst}' during {_current_phase()}", target=str(dst), culprit=_caller_info()
    )


def _guarded_shutil_copytree(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
    raise BlockedEffect(
        f"cannot copy directory tree to '{dst}' during {_current_phase()}",
        target=str(dst),
        culprit=_caller_info(),
    )


def _guarded_shutil_move(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
    raise BlockedEffect(
        f"cannot move '{src}' to '{dst}' during {_current_phase()}",
        target=str(dst),
        culprit=_caller_info(),
    )


def _guarded_shutil_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot remove directory tree '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_shutil_chown(path: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot chown '{path}' during {_current_phase()}", target=str(path), culprit=_caller_info()
    )


def _guarded_path_touch(self: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(self):
        return _orig_path_touch(self, *args, **kwargs)
    raise BlockedEffect(
        f"cannot touch '{self}' during {_current_phase()}", target=str(self), culprit=_caller_info()
    )


def _guarded_path_mkdir(self: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(self):
        return _orig_path_mkdir(self, *args, **kwargs)
    raise BlockedEffect(
        f"cannot make directory '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_rmdir(self: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot remove directory '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_unlink(self: Any, *args: Any, **kwargs: Any) -> None:
    if _is_allowed_write(self):
        return _orig_path_unlink(self, *args, **kwargs)
    raise BlockedEffect(
        f"cannot unlink '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_rename(self: Any, target: Any) -> Any:
    if _is_allowed_write(self) and _is_allowed_write(target):
        return _orig_path_rename(self, target)
    raise BlockedEffect(
        f"cannot rename '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_replace(self: Any, target: Any) -> Any:
    if _is_allowed_write(self) and _is_allowed_write(target):
        return _orig_path_replace(self, target)
    raise BlockedEffect(
        f"cannot replace '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_symlink_to(self: Any, target: Any, target_is_directory: bool = False) -> None:
    raise BlockedEffect(
        f"cannot symlink '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_hardlink_to(self: Any, target: Any) -> None:
    raise BlockedEffect(
        f"cannot hardlink '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_chmod(self: Any, mode: int, *, follow_symlinks: bool = True) -> None:
    raise BlockedEffect(
        f"cannot chmod '{self}' during {_current_phase()}", target=str(self), culprit=_caller_info()
    )


def _guarded_path_write_text(self: Any, data: str, *args: Any, **kwargs: Any) -> int:
    if _is_allowed_write(self):
        return _orig_path_write_text(self, data, *args, **kwargs)
    raise BlockedEffect(
        f"cannot write to '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_write_bytes(self: Any, data: bytes) -> int:
    if _is_allowed_write(self):
        return _orig_path_write_bytes(self, data)
    raise BlockedEffect(
        f"cannot write to '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_open(self: Any, *args: Any, **kwargs: Any) -> Any:
    mode = args[0] if args else kwargs.get("mode", "r")
    if isinstance(mode, str) and _is_write_mode(mode) and not _is_allowed_write(self):
        culprit = _caller_info()
        raise BlockedEffect(
            f"cannot open '{self}' for writing during {_current_phase()}",
            target=str(self),
            culprit=culprit,
        )
    return _orig_path_open(self, *args, **kwargs)


def _guarded_socket_connect(self: Any, address: Any) -> None:
    culprit = _caller_info()
    raise BlockedEffect(
        f"outbound network connection to {address!r} blocked during {_current_phase()}",
        target=str(address),
        culprit=culprit,
    )


def _guarded_socket_connect_ex(self: Any, address: Any) -> int:
    culprit = _caller_info()
    raise BlockedEffect(
        f"outbound network connection to {address!r} blocked during {_current_phase()}",
        target=str(address),
        culprit=culprit,
    )


def _guarded_socket_sendto(self: Any, *args: Any, **kwargs: Any) -> int:
    culprit = _caller_info()
    raise BlockedEffect(
        f"outbound network sendto blocked during {_current_phase()}",
        culprit=culprit,
    )


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
    culprit = _caller_info()
    raise BlockedEffect(
        f"outbound network connection to {address!r} blocked during {_current_phase()}",
        target=str(address),
        culprit=culprit,
    )


def _guarded_socket_getaddrinfo(*args: Any, **kwargs: Any) -> Any:
    host = args[0] if args else kwargs.get("host")
    culprit = _caller_info()
    target_str = str(host) if host is not None else None
    raise BlockedEffect(
        f"DNS resolution for {host!r} blocked during {_current_phase()}",
        target=target_str,
        culprit=culprit,
    )


def _guarded_socket_gethostbyname(hostname: Any, *args: Any, **kwargs: Any) -> Any:
    culprit = _caller_info()
    raise BlockedEffect(
        f"DNS resolution for {hostname!r} blocked during {_current_phase()}",
        target=str(hostname),
        culprit=culprit,
    )


def _guarded_socket_gethostbyname_ex(hostname: Any, *args: Any, **kwargs: Any) -> Any:
    culprit = _caller_info()
    raise BlockedEffect(
        f"DNS resolution for {hostname!r} blocked during {_current_phase()}",
        target=str(hostname),
        culprit=culprit,
    )


def _guarded_socket_gethostbyaddr(ip_address: Any, *args: Any, **kwargs: Any) -> Any:
    culprit = _caller_info()
    raise BlockedEffect(
        f"DNS resolution for {ip_address!r} blocked during {_current_phase()}",
        target=str(ip_address),
        culprit=culprit,
    )


def _guarded_socket_getnameinfo(sockaddr: Any, *args: Any, **kwargs: Any) -> Any:
    culprit = _caller_info()
    raise BlockedEffect(
        f"DNS resolution for {sockaddr!r} blocked during {_current_phase()}",
        target=str(sockaddr),
        culprit=culprit,
    )


def _guarded_process_spawn(*args: Any, **kwargs: Any) -> Any:
    cmd = args[0] if args else kwargs.get("args")
    if cmd is None:
        cmd = kwargs.get("cmd") or kwargs.get("command")
    culprit = _caller_info()
    msg = (
        f"cannot spawn process '{cmd}' during {_current_phase()}"
        if cmd is not None
        else f"cannot spawn process during {_current_phase()}"
    )
    raise BlockedEffect(msg, target=str(cmd) if cmd is not None else None, culprit=culprit)


class _GuardedPopen(_orig_subprocess_popen):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        cmd = args[0] if args else kwargs.get("args")
        culprit = _caller_info()
        msg = (
            f"cannot spawn process '{cmd}' during {_current_phase()}"
            if cmd is not None
            else f"cannot spawn process during {_current_phase()}"
        )
        raise BlockedEffect(msg, target=str(cmd) if cmd is not None else None, culprit=culprit)


def _guarded_os_system(command: Any) -> int:
    culprit = _caller_info()
    raise BlockedEffect(
        f"cannot spawn process '{command}' during {_current_phase()}",
        target=str(command),
        culprit=culprit,
    )


def _guarded_os_posix_spawn(path: Any, *args: Any, **kwargs: Any) -> int:
    culprit = _caller_info()
    raise BlockedEffect(
        f"cannot spawn process '{path}' during {_current_phase()}",
        target=str(path),
        culprit=culprit,
    )


def _guarded_os_spawn(mode: Any, file: Any, *args: Any, **kwargs: Any) -> Any:
    culprit = _caller_info()
    raise BlockedEffect(
        f"cannot spawn process '{file}' during {_current_phase()}",
        target=str(file),
        culprit=culprit,
    )


def _guarded_os_exec(file: Any, *args: Any, **kwargs: Any) -> None:
    culprit = _caller_info()
    raise BlockedEffect(
        f"cannot spawn process '{file}' during {_current_phase()}",
        target=str(file),
        culprit=culprit,
    )


_depth = 0


def _install_guard() -> None:
    builtins.open = _guarded_open
    io.open = _guarded_open
    if hasattr(builtins, "_io") and hasattr(builtins._io, "open"):
        builtins._io.open = _guarded_open

    os.open = _guarded_os_open
    os.remove = _guarded_os_remove
    os.unlink = _guarded_os_unlink
    os.rmdir = _guarded_os_rmdir
    os.removedirs = _guarded_os_removedirs
    os.mkdir = _guarded_os_mkdir
    os.makedirs = _guarded_os_makedirs
    os.rename = _guarded_os_rename
    os.renames = _guarded_os_renames
    os.replace = _guarded_os_replace
    os.link = _guarded_os_link
    os.symlink = _guarded_os_symlink
    os.chmod = _guarded_os_chmod
    os.chown = _guarded_os_chown
    if hasattr(os, "lchmod"):
        os.lchmod = _guarded_os_chmod
    if hasattr(os, "lchown"):
        os.lchown = _guarded_os_chown
    os.truncate = _guarded_os_truncate
    os.ftruncate = _guarded_os_ftruncate
    os.utime = _guarded_os_utime

    shutil.copy = _guarded_shutil_copy
    shutil.copy2 = _guarded_shutil_copy2
    shutil.copyfile = _guarded_shutil_copyfile
    shutil.copytree = _guarded_shutil_copytree
    shutil.move = _guarded_shutil_move
    shutil.rmtree = _guarded_shutil_rmtree
    shutil.chown = _guarded_shutil_chown

    pathlib.Path.touch = _guarded_path_touch
    pathlib.Path.mkdir = _guarded_path_mkdir
    pathlib.Path.rmdir = _guarded_path_rmdir
    pathlib.Path.unlink = _guarded_path_unlink
    pathlib.Path.rename = _guarded_path_rename
    pathlib.Path.replace = _guarded_path_replace
    pathlib.Path.symlink_to = _guarded_path_symlink_to
    pathlib.Path.hardlink_to = _guarded_path_hardlink_to
    pathlib.Path.chmod = _guarded_path_chmod
    if hasattr(pathlib.Path, "lchmod"):
        pathlib.Path.lchmod = _guarded_path_chmod
    pathlib.Path.write_text = _guarded_path_write_text
    pathlib.Path.write_bytes = _guarded_path_write_bytes
    pathlib.Path.open = _guarded_path_open

    socket.socket.connect = _guarded_socket_connect
    socket.socket.connect_ex = _guarded_socket_connect_ex
    socket.socket.sendto = _guarded_socket_sendto
    socket.create_connection = _guarded_create_connection
    socket.getaddrinfo = _guarded_socket_getaddrinfo
    socket.gethostbyname = _guarded_socket_gethostbyname
    socket.gethostbyname_ex = _guarded_socket_gethostbyname_ex
    socket.gethostbyaddr = _guarded_socket_gethostbyaddr
    socket.getnameinfo = _guarded_socket_getnameinfo

    subprocess.Popen = _GuardedPopen
    subprocess.Popen.__init__ = _GuardedPopen.__init__
    _orig_subprocess_popen.__init__ = _GuardedPopen.__init__
    subprocess.run = _guarded_process_spawn
    subprocess.call = _guarded_process_spawn
    subprocess.check_call = _guarded_process_spawn
    subprocess.check_output = _guarded_process_spawn
    if hasattr(subprocess, "getoutput"):
        subprocess.getoutput = _guarded_process_spawn
    if hasattr(subprocess, "getstatusoutput"):
        subprocess.getstatusoutput = _guarded_process_spawn

    os.system = _guarded_os_system
    if hasattr(os, "posix_spawn"):
        os.posix_spawn = _guarded_os_posix_spawn
    if hasattr(os, "posix_spawnp"):
        os.posix_spawnp = _guarded_os_posix_spawn
    for spawn_name in (
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
    ):
        if hasattr(os, spawn_name):
            setattr(os, spawn_name, _guarded_os_spawn)
    for exec_name in (
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
    ):
        if hasattr(os, exec_name):
            setattr(os, exec_name, _guarded_os_exec)


def _remove_guard() -> None:
    builtins.open = _orig_builtins_open
    io.open = _orig_io_open
    if hasattr(builtins, "_io") and hasattr(builtins._io, "open"):
        builtins._io.open = _orig_builtins_open

    os.open = _orig_os_open
    os.remove = _orig_os_remove
    os.unlink = _orig_os_unlink
    os.rmdir = _orig_os_rmdir
    os.removedirs = _orig_os_removedirs
    os.mkdir = _orig_os_mkdir
    os.makedirs = _orig_os_makedirs
    os.rename = _orig_os_rename
    os.renames = _orig_os_renames
    os.replace = _orig_os_replace
    os.link = _orig_os_link
    os.symlink = _orig_os_symlink
    os.chmod = _orig_os_chmod
    os.chown = _orig_os_chown
    if _orig_os_lchmod is not None:
        os.lchmod = _orig_os_lchmod
    if _orig_os_lchown is not None:
        os.lchown = _orig_os_lchown
    os.truncate = _orig_os_truncate
    os.ftruncate = _orig_os_ftruncate
    os.utime = _orig_os_utime

    shutil.copy = _orig_shutil_copy
    shutil.copy2 = _orig_shutil_copy2
    shutil.copyfile = _orig_shutil_copyfile
    shutil.copytree = _orig_shutil_copytree
    shutil.move = _orig_shutil_move
    shutil.rmtree = _orig_shutil_rmtree
    shutil.chown = _orig_shutil_chown

    pathlib.Path.touch = _orig_path_touch
    pathlib.Path.mkdir = _orig_path_mkdir
    pathlib.Path.rmdir = _orig_path_rmdir
    pathlib.Path.unlink = _orig_path_unlink
    pathlib.Path.rename = _orig_path_rename
    pathlib.Path.replace = _orig_path_replace
    pathlib.Path.symlink_to = _orig_path_symlink_to
    pathlib.Path.hardlink_to = _orig_path_hardlink_to
    pathlib.Path.chmod = _orig_path_chmod
    if _orig_path_lchmod is not None:
        pathlib.Path.lchmod = _orig_path_lchmod
    pathlib.Path.write_text = _orig_path_write_text
    pathlib.Path.write_bytes = _orig_path_write_bytes
    pathlib.Path.open = _orig_path_open

    socket.socket.connect = _orig_socket_connect
    socket.socket.connect_ex = _orig_socket_connect_ex
    socket.socket.sendto = _orig_socket_sendto
    socket.create_connection = _orig_create_connection
    socket.getaddrinfo = _orig_socket_getaddrinfo
    socket.gethostbyname = _orig_socket_gethostbyname
    socket.gethostbyname_ex = _orig_socket_gethostbyname_ex
    socket.gethostbyaddr = _orig_socket_gethostbyaddr
    socket.getnameinfo = _orig_socket_getnameinfo

    subprocess.Popen = _orig_subprocess_popen
    subprocess.Popen.__init__ = _orig_subprocess_popen_init
    _orig_subprocess_popen.__init__ = _orig_subprocess_popen_init
    subprocess.run = _orig_subprocess_run
    subprocess.call = _orig_subprocess_call
    subprocess.check_call = _orig_subprocess_check_call
    subprocess.check_output = _orig_subprocess_check_output
    if _orig_subprocess_getoutput is not None:
        subprocess.getoutput = _orig_subprocess_getoutput
    if _orig_subprocess_getstatusoutput is not None:
        subprocess.getstatusoutput = _orig_subprocess_getstatusoutput

    os.system = _orig_os_system
    if _orig_os_posix_spawn is not None:
        os.posix_spawn = _orig_os_posix_spawn
    if _orig_os_posix_spawnp is not None:
        os.posix_spawnp = _orig_os_posix_spawnp
    if _orig_os_spawnl is not None:
        os.spawnl = _orig_os_spawnl
    if _orig_os_spawnle is not None:
        os.spawnle = _orig_os_spawnle
    if _orig_os_spawnlp is not None:
        os.spawnlp = _orig_os_spawnlp
    if _orig_os_spawnlpe is not None:
        os.spawnlpe = _orig_os_spawnlpe
    if _orig_os_spawnv is not None:
        os.spawnv = _orig_os_spawnv
    if _orig_os_spawnve is not None:
        os.spawnve = _orig_os_spawnve
    if _orig_os_spawnvp is not None:
        os.spawnvp = _orig_os_spawnvp
    if _orig_os_spawnvpe is not None:
        os.spawnvpe = _orig_os_spawnvpe

    if _orig_os_execl is not None:
        os.execl = _orig_os_execl
    if _orig_os_execle is not None:
        os.execle = _orig_os_execle
    if _orig_os_execlp is not None:
        os.execlp = _orig_os_execlp
    if _orig_os_execlpe is not None:
        os.execlpe = _orig_os_execlpe
    if _orig_os_execv is not None:
        os.execv = _orig_os_execv
    if _orig_os_execve is not None:
        os.execve = _orig_os_execve
    if _orig_os_execvp is not None:
        os.execvp = _orig_os_execvp
    if _orig_os_execvpe is not None:
        os.execvpe = _orig_os_execvpe


@contextmanager
def active_guard(enabled: bool = True, phase: str = "import", protect: Any = ()):
    """Activate the effect guard within a context block.

    `protect` names trees that stay blocked even under the scratch allowance:
    the project being checked lives there, so a checkpoint or source overwrite
    inside it is exactly what the guard is for, wherever that tree happens to
    sit. Nested contexts union, and the roots apply whether or not this
    particular context enables the hooks, so a caller can protect a whole job
    with `active_guard(enabled=False, protect=[root])`.
    """
    global _depth
    _protected_stack.append(_resolve_roots(protect))
    try:
        if not enabled:
            yield
            return
        _contexts.append(phase)
        if _depth == 0:
            _install_guard()
        _depth += 1
        try:
            yield
        finally:
            _depth -= 1
            if _depth == 0:
                _remove_guard()
            _contexts.pop()
    finally:
        _protected_stack.pop()
