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
import inspect
import io
import os
import pathlib
import shutil
import socket
import tempfile
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
    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, BlockedEffect):
            return cur
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


def _is_allowed_write(target: Any) -> bool:
    # Importing torch and tracing on meta tensors populates the inductor and
    # triton caches. Those writes come from us, not from the user's code, and
    # blocking them would break the check we are trying to run.
    try:
        p = Path(target)
        for part in p.parts:
            if part.startswith(("torchinductor_", "triton_")):
                return True
    except (TypeError, ValueError, OSError):
        return False
    return False


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
    raise BlockedEffect(
        f"cannot remove '{path}' during {_current_phase()}",
        target=str(path),
        culprit=_caller_info(),
    )


def _guarded_os_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
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
    raise BlockedEffect(
        f"cannot rename '{src}' during {_current_phase()}", target=str(src), culprit=_caller_info()
    )


def _guarded_os_renames(old: Any, new: Any, *args: Any, **kwargs: Any) -> None:
    raise BlockedEffect(
        f"cannot rename '{old}' during {_current_phase()}", target=str(old), culprit=_caller_info()
    )


def _guarded_os_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
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
    raise BlockedEffect(
        f"cannot unlink '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_rename(self: Any, target: Any) -> Any:
    raise BlockedEffect(
        f"cannot rename '{self}' during {_current_phase()}",
        target=str(self),
        culprit=_caller_info(),
    )


def _guarded_path_replace(self: Any, target: Any) -> Any:
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


@contextmanager
def active_guard(enabled: bool = True, phase: str = "import"):
    """Activate the effect guard within a context block."""
    global _depth
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
