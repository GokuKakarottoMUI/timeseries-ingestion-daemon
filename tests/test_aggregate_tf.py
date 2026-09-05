"""Custom-timeframe aggregation.

Non-standard intervals are built from an already-stored base timeframe instead
of being re-fetched, so the aggregation has to reproduce exactly what the
exchange would have returned: open of the first candle, close of the last, max
high, min low, summed volume — per slot.

These tests build ``2d`` from ``1d``. Slots for intervals longer than a day are
anchored at 1 Jan UTC of the year.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

import ingestion.calculate_tf_and_custom_tf as ctf
from ingestion.calculate_tf_and_custom_tf import Calculate_Tf_And_CustomTF

DAY_MS = 86_400_000
YEAR_START = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

# Custom-timeframe definitions owned by the tests. The shipped JSON is runtime
# state an operator toggles from the config GUI; a test that reads it would pass
# or fail depending on a checkbox, which says nothing about the aggregation.
TEST_CUSTOM_TIMEFRAMES = {
    "enable": True,
    "custom_intervals": {
        "2d": {"active": True, "active_featured": True,
               "minutes": 2880, "hours": 48, "source": "1d"},
        "8d": {"active": False, "active_featured": False,
               "minutes": 11520, "hours": 192, "source": "1d"},
    },
}


@pytest.fixture(autouse=True)
def pinned_custom_timeframes(monkeypatch):
    monkeypatch.setattr(ctf, "CUSTOM_TIMEFRAMES", TEST_CUSTOM_TIMEFRAMES)


@pytest.fixture
def calc():
    return Calculate_Tf_And_CustomTF()


def make_daily(n, first_ts=YEAR_START):
    """n daily candles with values that make each aggregation rule distinguishable.

    Day i: open=100+i, high=200+i, low=50-i, close=150+i, volume=10+i.
    Highs rise and lows fall with i, so max/min pick a *known* member of each slot.
    """
    arr = np.empty((n, 6), dtype=np.float64)
    arr[:, 0] = first_ts + np.arange(n) * DAY_MS
    arr[:, 1] = 100.0 + np.arange(n)
    arr[:, 2] = 200.0 + np.arange(n)
    arr[:, 3] = 50.0 - np.arange(n)
    arr[:, 4] = 150.0 + np.arange(n)
    arr[:, 5] = 10.0 + np.arange(n)
    return arr


def test_two_daily_candles_become_one_2d_candle(calc):
    out = calc._aggregate_candles_batch_numpy(make_daily(2), "1d", "2d")

    assert out.shape == (1, 6)
    ts, o, h, l, c, v = out[0]
    assert ts == YEAR_START          # slot anchored at the year boundary
    assert o == 100.0                # open of the FIRST candle in the slot
    assert c == 151.0                # close of the LAST candle in the slot
    assert h == 201.0                # max high (day 1)
    assert l == 49.0                 # min low  (day 1)
    assert v == 21.0                 # 10 + 11


def test_slots_are_independent_and_contiguous(calc):
    out = calc._aggregate_candles_batch_numpy(make_daily(6), "1d", "2d")

    assert out.shape == (3, 6)
    np.testing.assert_array_equal(
        out[:, 0], [YEAR_START, YEAR_START + 2 * DAY_MS, YEAR_START + 4 * DAY_MS]
    )
    np.testing.assert_allclose(out[:, 1], [100, 102, 104])   # opens
    np.testing.assert_allclose(out[:, 4], [151, 153, 155])   # closes
    np.testing.assert_allclose(out[:, 2], [201, 203, 205])   # highs
    np.testing.assert_allclose(out[:, 3], [49, 47, 45])      # lows
    np.testing.assert_allclose(out[:, 5], [21, 25, 29])      # volumes


def test_output_is_a_fresh_contiguous_float64_block(calc):
    out = calc._aggregate_candles_batch_numpy(make_daily(6), "1d", "2d")
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]


def test_incomplete_trailing_slot_is_not_emitted(calc):
    """A 2d slot needs both of its days. Five daily candles give two complete
    slots; the dangling fifth day must wait rather than produce a half candle."""
    out = calc._aggregate_candles_batch_numpy(make_daily(5), "1d", "2d")
    assert out.shape == (2, 6)
    assert out[-1, 0] == YEAR_START + 2 * DAY_MS


def test_gap_in_source_still_aggregates_the_days_present(calc):
    """A missing day must not shift later candles into the wrong slot: slot
    membership comes from the timestamp, not from position in the array."""
    full = make_daily(6)
    with_gap = np.delete(full, 2, axis=0)      # drop day 2 (first day of slot 2)

    out = calc._aggregate_candles_batch_numpy(with_gap, "1d", "2d")

    assert out.shape == (3, 6)
    np.testing.assert_array_equal(
        out[:, 0], [YEAR_START, YEAR_START + 2 * DAY_MS, YEAR_START + 4 * DAY_MS]
    )
    # slot 2 now holds only day 3 → its aggregate is that single candle
    np.testing.assert_allclose(out[1], [YEAR_START + 2 * DAY_MS, 103, 203, 47, 153, 13])


def test_duplicate_timestamps_are_collapsed(calc):
    """The source may hold the same candle twice after a retry; aggregation
    must not double-count its volume."""
    dup = np.vstack([make_daily(2), make_daily(2)[:1]])
    intervals = np.array([[YEAR_START, YEAR_START + 10 * DAY_MS]], dtype=np.int64)

    out = calc._aggregate_candles_batch_numpy(dup, "1d", "2d", intervals)

    assert out.shape == (1, 6)
    assert out[0, 5] == 21.0


def test_empty_source_returns_an_empty_batch(calc):
    out = calc._aggregate_candles_batch_numpy(
        np.empty((0, 6), dtype=np.float64), "1d", "2d"
    )
    assert out.shape == (0, 6)


def test_inactive_target_timeframe_produces_nothing(calc):
    """`8d` exists in the config but is active=false — it must not be built."""
    out = calc._aggregate_candles_batch_numpy(make_daily(30), "1d", "8d")
    assert out.shape == (0, 6)


@pytest.mark.parametrize("name,minutes", [("1h", 60), ("1d", 1440), ("2d", 2880)])
def test_timeframe_minutes_resolve_for_both_standard_and_custom(calc, name, minutes):
    assert calc._get_timeframe_minutes(name) == minutes


def test_unknown_timeframe_resolves_to_zero(calc):
    assert calc._get_timeframe_minutes("42x") == 0
