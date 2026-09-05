"""Per-array reader/writer lock (POSIX ``flock``) coordinating READ vs CONSOLIDATE/VACUUM across processes.

Why it exists: the writer process (``continuous_fetch``) runs ``tiledb.consolidate`` +
``tiledb.vacuum``, which physically DELETE fragment files, while the reader process
(``get_data``) has the same array open. TileDB reads fragment files lazily, so the window
between ``tiledb.open`` (which snapshots the fragment list) and the actual slice
``array[tf, ts:]`` can overlap a vacuum → ``ENOENT`` → the reader silently drops one
(symbol, timeframe) pair. ``flock`` closes that window: the reader holds SHARED for a whole
symbol read, the writer holds EXCLUSIVE around consolidate+vacuum, so a vacuum waits for
in-flight readers and blocks new ones until it finishes.

Properties:
  * C core — ``fcntl`` is a CPython built-in C module calling the ``flock(2)`` syscall directly.
  * Crash-safe — the lock lives on the file descriptor, so closing it (or the process dying)
    releases it; no stale locks to clean up.
  * Deadlock-free — a process holds at most one array lock at a time, so no wait cycle.
  * Local storage only (``flock`` is standard POSIX advisory locking; not valid over NFS).

Reader and writer import THIS module, so the lock-file naming has a single definition.
"""
from __future__ import annotations
import os
import fcntl
from contextlib import contextmanager


def _lock_file(array_path: str) -> str:
    """Path of the lock file for one array: ``{data_root}/.rwlocks/{mc}__{sc}__{sym}.lock``.

    ``array_path`` is ``{data_root}/{group}/{mc}/{sc}/{sym}``, so ``{data_root}`` is four
    ``dirname`` levels up. The root is derived STRUCTURALLY rather than by matching the group
    name, because the enclosing path may itself contain that name. The lock file is named from
    the last three levels (unique per symbol), so reader and writer given the same
    ``array_path`` always land on the same file. ``.rwlocks`` sits NEXT TO the group root, never
    inside it — a stray file inside a TileDB group or array would confuse TileDB.
    """
    ap = array_path.rstrip(os.sep)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ap))))
    d = os.path.join(root, ".rwlocks")
    os.makedirs(d, exist_ok=True)
    name = "__".join(ap.split(os.sep)[-3:])
    return os.path.join(d, name + ".lock")


@contextmanager
def read_lock(array_path: str):
    """flock SHARED quanh đọc 1 array — nhiều reader song song OK; CHỜ nếu writer đang EXCLUSIVE (vacuum)."""
    f = open(_lock_file(array_path), "a")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        yield
    finally:
        f.close()   # đóng fd ⇒ OS tự nhả flock (kể cả khi exception)


@contextmanager
def write_lock(array_path: str):
    """flock EXCLUSIVE quanh consolidate+vacuum — CHỜ reader đang đọc xong + chặn reader mới tới khi xong."""
    f = open(_lock_file(array_path), "a")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        f.close()
