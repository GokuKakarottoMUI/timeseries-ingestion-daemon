"""Cross-process reader/writer locking.

The bug this prevents: TileDB ``vacuum`` deletes fragment files while a reader
holds a stale fragment list, and the reader loses a (symbol, timeframe) pair
silently. The contract that has to hold is that reader and writer, given the same
array path, contend on the same file — and that the lock file never lands inside
the TileDB tree.
"""
import fcntl
import os
import threading

import pytest

from ingestion.config_fetch_data import DATABASE_ROOT_PATH, build_array_path
from ingestion.rwlock import _lock_file, read_lock, write_lock

ARRAY = build_array_path("Cryptocurrency", "BTC", "BTCUSD")


def test_reader_and_writer_derive_the_same_lock_file():
    """Reader and writer are different processes running different code; if they
    disagreed on the path the lock would be a no-op."""
    assert _lock_file(ARRAY) == _lock_file(ARRAY)


def test_trailing_separator_does_not_split_the_lock():
    assert _lock_file(ARRAY + os.sep) == _lock_file(ARRAY)


def test_lock_file_sits_beside_the_group_not_inside_it():
    """A stray file inside a TileDB group or array directory confuses TileDB, so
    .rwlocks must be a sibling of the group root."""
    path = _lock_file(ARRAY)
    assert os.path.dirname(path) == os.path.join(DATABASE_ROOT_PATH, ".rwlocks")
    assert not path.startswith(os.path.join(DATABASE_ROOT_PATH, "market_data") + os.sep)


def test_lock_name_is_unique_per_symbol():
    a = _lock_file(build_array_path("Cryptocurrency", "BTC", "BTCUSD"))
    b = _lock_file(build_array_path("Cryptocurrency", "ETH", "ETHUSD"))
    c = _lock_file(build_array_path("Equities", "BTC", "BTCUSD"))
    assert len({a, b, c}) == 3
    assert os.path.basename(a) == "Cryptocurrency__BTC__BTCUSD.lock"


def test_readers_do_not_block_each_other():
    """Multiple training readers must run concurrently; SHARED is not a mutex."""
    with read_lock(ARRAY):
        with read_lock(ARRAY):
            assert True


def test_writer_is_excluded_while_a_reader_holds_the_lock():
    """flock is per open-file-description, so a second open() contends even from
    the same process — which is exactly what a separate reader process does."""
    with read_lock(ARRAY):
        probe = open(_lock_file(ARRAY), "a")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            probe.close()


def test_reader_is_excluded_while_the_writer_holds_the_lock():
    with write_lock(ARRAY):
        probe = open(_lock_file(ARRAY), "a")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        finally:
            probe.close()


def test_lock_is_released_when_the_body_raises():
    """The lock lives on the fd, so an exception inside consolidate must not
    leave every reader blocked forever."""
    with pytest.raises(RuntimeError):
        with write_lock(ARRAY):
            raise RuntimeError("vacuum blew up")

    probe = open(_lock_file(ARRAY), "a")
    try:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)   # must not raise
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    finally:
        probe.close()


def test_writer_waits_for_an_in_flight_reader():
    """End-to-end ordering: a writer that arrives mid-read finishes after it."""
    order = []
    reader_holds = threading.Event()
    writer_done = threading.Event()

    def writer():
        reader_holds.wait(timeout=5)
        with write_lock(ARRAY):
            order.append("write")
        writer_done.set()

    t = threading.Thread(target=writer)
    with read_lock(ARRAY):
        t.start()
        reader_holds.set()
        assert not writer_done.wait(timeout=0.3), "writer entered while a reader held SHARED"
        order.append("read")

    t.join(timeout=5)
    assert order == ["read", "write"]
