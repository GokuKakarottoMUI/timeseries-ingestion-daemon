"""End-to-end storage round-trip against a real TileDB array.

conftest points the data root at a temporary directory, so these tests create
and write actual arrays without touching any real database.
"""
import numpy as np
import pytest
import tiledb

from ingestion.calculate_tf_and_custom_tf import Calculate_Tf_And_CustomTF
from ingestion.config_fetch_data import build_array_path
from ingestion.database import DatabaseManager
from get_data.get_data_from_database import BASE_COLS

HOUR_MS = 3_600_000
START_MS = 1735689600000   # 2025-01-01T00:00:00Z

OHLCV_ATTRS = ("open", "high", "low", "close", "volume")


@pytest.fixture
def manager():
    return DatabaseManager(Calculate_Tf_And_CustomTF())


@pytest.fixture
def array_path(manager, tmp_path):
    path = str(tmp_path / "BTCUSD")
    manager.create_symbol_array(path)
    return path


def make_candles(n, first_ts=START_MS, step=HOUR_MS):
    """(N, 6) float64 [timestamp, open, high, low, close, volume], C-ordered —
    exactly what the fetch layer hands to the writer."""
    arr = np.empty((n, 6), dtype=np.float64)
    arr[:, 0] = first_ts + np.arange(n) * step
    arr[:, 1] = 93000.0 + np.arange(n)          # open
    arr[:, 2] = 93500.0 + np.arange(n)          # high
    arr[:, 3] = 92500.0 + np.arange(n)          # low
    arr[:, 4] = 93250.0 + np.arange(n)          # close
    arr[:, 5] = 1.5 + np.arange(n)              # volume
    return arr


# ── Schema ────────────────────────────────────────────────────────────────────

def test_schema_stores_ohlcv_and_nothing_else(array_path):
    """The stored attributes are exactly OHLCV. Anything extra costs disk and
    read bandwidth on every query, so the set is pinned here deliberately."""
    with tiledb.open(array_path) as arr:
        names = tuple(arr.schema.attr(i).name for i in range(arr.schema.nattr))
    assert names == OHLCV_ATTRS


def test_every_attribute_is_float64(array_path):
    with tiledb.open(array_path) as arr:
        for i in range(arr.schema.nattr):
            assert arr.schema.attr(i).dtype == np.float64


def test_dimensions_are_timeframe_and_timestamp(array_path):
    with tiledb.open(array_path) as arr:
        dom = arr.schema.domain
        assert dom.dim(0).name == "timeframe_minutes"
        assert dom.dim(0).dtype == np.int32
        assert dom.dim(1).name == "timestamp"
        assert dom.dim(1).dtype == np.int64
        assert arr.schema.sparse
        assert not arr.schema.allows_duplicates


def test_read_path_column_order_matches_the_schema(array_path):
    """BASE_COLS drives the shape of the assembled block, so it must stay in
    step with the attributes actually stored."""
    with tiledb.open(array_path) as arr:
        stored = tuple(arr.schema.attr(i).name for i in range(arr.schema.nattr))
    assert BASE_COLS == stored


# ── Write / read round-trip ───────────────────────────────────────────────────

def test_values_survive_the_round_trip_exactly(manager, array_path):
    candles = make_candles(48)
    written = manager._write_single(
        array_path, 60, candles, "BTCUSD", "1h", candles[:, 0].astype(np.int64)
    )
    assert written == 48

    out = manager.query_candles(array_path, 60, START_MS, START_MS + 48 * HOUR_MS)

    assert out.shape == (48, 6)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, candles)


def test_written_columns_are_contiguous_for_tiledb(manager, array_path):
    """The write path converts a C-ordered batch once so each column buffer is
    contiguous. If that regressed, TileDB would silently copy per column."""
    candles = make_candles(10)
    assert not candles[:, 1].flags["C_CONTIGUOUS"], "fixture must feed a C-ordered batch"

    cols = np.asfortranarray(candles[:, 1:6])
    assert all(cols[:, j].flags["C_CONTIGUOUS"] for j in range(5))
    np.testing.assert_array_equal(cols, candles[:, 1:6])


def test_timeframes_share_one_array_without_mixing(manager, array_path):
    """All timeframes of a symbol live in the same array; a query for one must
    not return another's candles."""
    hourly = make_candles(24, step=HOUR_MS)
    daily = make_candles(3, step=24 * HOUR_MS)
    manager._write_single(array_path, 60, hourly, "BTCUSD", "1h",
                          hourly[:, 0].astype(np.int64))
    manager._write_single(array_path, 1440, daily, "BTCUSD", "1d",
                          daily[:, 0].astype(np.int64))

    got_hourly = manager.query_candles(array_path, 60, 0, START_MS + 400 * HOUR_MS)
    got_daily = manager.query_candles(array_path, 1440, 0, START_MS + 400 * HOUR_MS)

    assert len(got_hourly) == 24
    assert len(got_daily) == 3
    np.testing.assert_array_equal(got_daily, daily)


def test_results_come_back_ordered_by_timestamp(manager, array_path):
    """cell_order='row-major' means TileDB returns candles sorted, so the read
    path never sorts. Feeding a shuffled batch must not break that."""
    candles = make_candles(30)
    shuffled = candles[np.random.default_rng(0).permutation(30)]
    manager._write_single(array_path, 60, shuffled, "BTCUSD", "1h",
                          shuffled[:, 0].astype(np.int64))

    out = manager.query_candles(array_path, 60, 0, START_MS + 100 * HOUR_MS)
    assert np.all(np.diff(out[:, 0]) > 0)


def test_query_outside_the_stored_range_is_empty_not_an_error(manager, array_path):
    candles = make_candles(5)
    manager._write_single(array_path, 60, candles, "BTCUSD", "1h",
                          candles[:, 0].astype(np.int64))

    out = manager.query_candles(array_path, 60, START_MS - 100 * HOUR_MS, START_MS - 1)
    assert out.shape == (0, 6)


def test_missing_array_is_reported_as_empty(manager, tmp_path):
    out = manager.query_candles(str(tmp_path / "nope"), 60, 0, START_MS)
    assert out.shape == (0, 6)


# ── batch_insert_data: dedup, empty input, path resolution ────────────────────

def test_batch_insert_writes_through_build_array_path(manager):
    """batch_insert_data resolves its own path; it must match the helper the
    reader uses, or writer and reader would target different directories."""
    candles = make_candles(12)
    written = manager.batch_insert_data("Cryptocurrency", "BTC", "RT_PATH", "1h", candles)
    assert written == 12

    path = build_array_path("Cryptocurrency", "BTC", "RT_PATH")
    out = manager.query_candles(path, 60, 0, START_MS + 100 * HOUR_MS)
    np.testing.assert_array_equal(out, candles)


def test_duplicate_timestamps_are_dropped_before_writing(manager):
    """allows_duplicates=False, so an un-deduped batch would make TileDB throw.
    A retried fetch that returns overlapping candles must still write."""
    candles = make_candles(6)
    with_dups = np.vstack([candles, candles[:3]])

    written = manager.batch_insert_data("Cryptocurrency", "BTC", "RT_DEDUP", "1h", with_dups)
    assert written == 6

    path = build_array_path("Cryptocurrency", "BTC", "RT_DEDUP")
    out = manager.query_candles(path, 60, 0, START_MS + 100 * HOUR_MS)
    assert len(out) == 6
    assert len(np.unique(out[:, 0])) == 6


def test_empty_batch_writes_nothing(manager):
    assert manager.batch_insert_data(
        "Cryptocurrency", "BTC", "RT_EMPTY", "1h", np.empty((0, 6), dtype=np.float64)
    ) == 0


def test_unknown_timeframe_is_refused(manager):
    assert manager.batch_insert_data(
        "Cryptocurrency", "BTC", "RT_BADTF", "42x", make_candles(3)
    ) == 0


def test_ms_epoch_timestamps_survive_the_float64_hop(manager, array_path):
    """Timestamps travel as float64 in the candle array and become int64 on the
    dimension. ms-epoch is ~1.7e12, well inside float64's exact-integer range,
    but a regression to float32 would corrupt it — assert exactness."""
    candles = make_candles(24)
    manager._write_single(array_path, 60, candles, "BTCUSD", "1h",
                          candles[:, 0].astype(np.int64))

    out = manager.query_candles(array_path, 60, 0, START_MS + 100 * HOUR_MS)
    np.testing.assert_array_equal(out[:, 0].astype(np.int64), candles[:, 0].astype(np.int64))


def test_cent_level_price_differences_are_preserved(manager, array_path):
    """The reason the pipeline is float64 end to end: at BTC price levels the
    float32 ulp exceeds a 0.01 tick, collapsing distinct prices."""
    candles = make_candles(2)
    candles[0, 4] = 131072.01
    candles[1, 4] = 131072.02
    manager._write_single(array_path, 60, candles, "BTCUSD", "1h",
                          candles[:, 0].astype(np.int64))

    out = manager.query_candles(array_path, 60, 0, START_MS + 10 * HOUR_MS)
    assert out[0, 4] != out[1, 4]
    assert out[1, 4] - out[0, 4] == pytest.approx(0.01, abs=1e-9)
    # the same two values are indistinguishable in float32
    assert np.float32(131072.01) == np.float32(131072.02)


# ── consolidate ───────────────────────────────────────────────────────────────

def test_consolidate_merges_fragments_without_losing_data(manager, array_path):
    """Each write makes a fragment; consolidate+vacuum must compact them and
    leave every candle readable."""
    for i in range(4):
        chunk = make_candles(6, first_ts=START_MS + i * 6 * HOUR_MS)
        manager._write_single(array_path, 60, chunk, "BTCUSD", "1h",
                              chunk[:, 0].astype(np.int64))

    before = manager.query_candles(array_path, 60, 0, START_MS + 100 * HOUR_MS)
    assert len(before) == 24

    manager.consolidate_array(array_path)

    after = manager.query_candles(array_path, 60, 0, START_MS + 100 * HOUR_MS)
    np.testing.assert_array_equal(after, before)


def test_consolidating_a_missing_array_is_a_no_op(manager, tmp_path):
    manager.consolidate_array(str(tmp_path / "nope"))   # must not raise
