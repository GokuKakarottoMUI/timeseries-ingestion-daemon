"""Assembly of the feature block, and the contract ``run()`` hands downstream.

The pipeline itself does no maths — the engines are tested separately. What can
break here is assembly: columns landing in the wrong order, the stored candles
being altered on the way through, or the row alignment between features and
timestamps drifting.
"""
import numpy as np
import pytest

from get_data.get_data_from_database import BASE_COLS
from indicators.config.config_indicators import COMPUTE_CONFIG, INDICATORS_CONFIG
from indicators.feature_pipeline import (
    build_engines,
    compute_block,
    feature_names,
    run,
)

N_INDICATOR_COLS = 13          # 5 EMA + 3 MACD + 2 RSI + 3 BB
N_FEATURE_COLS = len(BASE_COLS) + N_INDICATOR_COLS


@pytest.fixture
def block():
    """A plausible OHLCV block in the exact layout get_data returns."""
    rng = np.random.default_rng(7)
    n = 400
    close = 90_000.0 + np.cumsum(rng.normal(0.0, 200.0, size=n))
    out = np.empty((n, len(BASE_COLS)), dtype=np.float64, order="F")
    out[:, 0] = close - 50.0        # open
    out[:, 1] = close + 120.0       # high
    out[:, 2] = close - 120.0       # low
    out[:, 3] = close
    out[:, 4] = rng.uniform(1.0, 50.0, size=n)
    return out


class FakeReader:
    """Stands in for GetDataFromDatabase, returning one prepared block."""

    def __init__(self, block, n_timeframes=1):
        self.block = block
        self.n_timeframes = n_timeframes

    def query_data_for_training(self):
        n = self.block.shape[0]
        entry = {"block": self.block, "columns": BASE_COLS,
                 "timestamp": np.arange(n, dtype=np.int64) * 86_400_000}
        for j, name in enumerate(BASE_COLS):
            entry[name] = self.block[:, j]
        tfs = {f"tf{i}": entry for i in range(self.n_timeframes)}
        return {"Cryptocurrency": {"BTC": {"BTCUSD": tfs}}}


# ── Column contract ──────────────────────────────────────────────────────────

def test_names_start_with_the_stored_columns():
    names = feature_names(build_engines())
    assert names[:len(BASE_COLS)] == BASE_COLS
    assert len(names) == N_FEATURE_COLS
    assert len(set(names)) == len(names), "duplicate column name"


def test_indicator_names_follow_configuration():
    names = feature_names(build_engines())[len(BASE_COLS):]
    assert names == (
        "ema_9", "ema_26", "ema_50", "ema_100", "ema_200",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "rsi_ma_14",
        "bb_basis", "bb_upper", "bb_lower",
    )


def test_column_names_track_the_config_not_the_code():
    """Renaming a length in YAML must rename the column, or config is a lie."""
    cfg = {
        "ema": {"lengths": [7]},
        "macd": {"fast": 3, "slow": 5, "signal": 2},
        "rsi": {"length": 21, "ma_length": 5},
        "bbands": {"length": 10, "mult": 1.5},
    }
    names = feature_names(build_engines(cfg))
    assert "ema_7" in names and "rsi_21" in names and "rsi_ma_5" in names
    assert "ema_9" not in names


# ── Block assembly ───────────────────────────────────────────────────────────

def test_stored_candles_pass_through_bit_for_bit(block):
    """The OHLCV prefix must be the input, unmodified — atol=0, not 'close'."""
    feat = compute_block(block, block[:, 3], build_engines())
    np.testing.assert_array_equal(feat[:, :len(BASE_COLS)], block)


def test_block_shape_dtype_and_layout(block):
    feat = compute_block(block, block[:, 3], build_engines())
    assert feat.shape == (block.shape[0], N_FEATURE_COLS)
    assert feat.dtype == np.float64
    assert feat.flags["F_CONTIGUOUS"], "columns must stay contiguous"


def test_compute_block_does_not_mutate_its_input(block):
    before = block.copy()
    compute_block(block, block[:, 3], build_engines())
    np.testing.assert_array_equal(block, before)


def test_indicator_columns_match_running_the_engines_alone(block):
    """Assembly must not shift or transpose anything on the way in."""
    feat = compute_block(block, block[:, 3], build_engines())
    col = len(BASE_COLS)
    for engine in build_engines():
        width = len(engine.names)
        direct = np.empty((block.shape[0], width), dtype=np.float64, order="F")
        engine.compute(block[:, 3], direct)
        np.testing.assert_allclose(feat[:, col:col + width], direct, rtol=1e-12)
        col += width


def test_output_dtype_is_honoured(block):
    feat = compute_block(block, block[:, 3], build_engines(), dtype=np.float32)
    assert feat.dtype == np.float32


# ── NaN policy ───────────────────────────────────────────────────────────────

def test_warmup_is_nan_and_everything_after_it_is_finite(block):
    """NaN elimination is deliberately absent: warmup stays NaN, and there is
    no NaN or Inf hiding after a column has started."""
    feat = compute_block(block, block[:, 3], build_engines())
    names = feature_names(build_engines())

    for j, name in enumerate(names[len(BASE_COLS):], start=len(BASE_COLS)):
        col = feat[:, j]
        finite = np.flatnonzero(np.isfinite(col))
        assert finite.size, f"{name} produced nothing"
        start = finite[0]
        assert np.all(np.isnan(col[:start])), f"{name} warmup is not NaN"
        assert np.all(np.isfinite(col[start:])), f"{name} has a hole after warmup"


def test_short_history_yields_nan_columns_rather_than_an_error(block):
    """A symbol with 30 candles cannot have an EMA-200; that must not raise."""
    feat = compute_block(block[:30], block[:30, 3], build_engines())
    names = feature_names(build_engines())
    assert feat.shape == (30, N_FEATURE_COLS)
    assert np.all(np.isnan(feat[:, names.index("ema_200")]))
    assert np.any(np.isfinite(feat[:, names.index("ema_9")]))


# ── run() ────────────────────────────────────────────────────────────────────

def test_run_keys_identify_symbol_and_timeframe(block):
    result = run(query=FakeReader(block, n_timeframes=2))
    assert set(result) == {
        ("Cryptocurrency", "BTC", "BTCUSD", "tf0"),
        ("Cryptocurrency", "BTC", "BTCUSD", "tf1"),
    }


def test_run_keeps_features_row_aligned_with_timestamps(block):
    entry = run(query=FakeReader(block))[("Cryptocurrency", "BTC", "BTCUSD", "tf0")]

    assert entry["feat"].shape == (block.shape[0], N_FEATURE_COLS)
    assert entry["timestamp"].shape == (block.shape[0],)
    assert entry["timestamp"].dtype == np.int64
    assert len(entry["names"]) == entry["feat"].shape[1]
    np.testing.assert_array_equal(entry["feat"][:, :len(BASE_COLS)], block)


def test_run_on_an_empty_database_returns_empty(block):
    class Empty:
        def query_data_for_training(self):
            return {}

    assert run(query=Empty()) == {}


# ── Configuration ────────────────────────────────────────────────────────────

def test_shipped_config_declares_every_parameter_the_engines_read():
    """Structure, not values: the lengths are meant to be tuned, but a missing
    key would fail at run time with a KeyError deep inside an engine."""
    assert INDICATORS_CONFIG["ema"]["lengths"], "at least one EMA length"
    assert all(int(v) > 0 for v in INDICATORS_CONFIG["ema"]["lengths"])
    assert {"fast", "slow", "signal"} <= set(INDICATORS_CONFIG["macd"])
    assert {"length", "ma_length"} <= set(INDICATORS_CONFIG["rsi"])
    assert {"length", "mult"} <= set(INDICATORS_CONFIG["bbands"])
    assert np.dtype(COMPUTE_CONFIG["output_dtype"]).kind == "f"


def test_shipped_config_builds_engines_without_error():
    """Whatever the operator has tuned it to, the shipped config must load."""
    engines = build_engines()
    names = feature_names(engines)
    assert names[:len(BASE_COLS)] == BASE_COLS
    assert len(names) > len(BASE_COLS)
