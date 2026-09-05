"""Pine parity for the indicator primitives and engines.

The reference values here are computed with explicit Python loops — slow, but
transparently a direct transcription of the Pine semantics, which is exactly
what a vectorized implementation needs to be checked against.

The warmup indices asserted below are the most fragile part of the port: an
off-by-one there produces a curve that looks plausible and disagrees with the
chart everywhere.
"""
import numpy as np
import pytest

from indicators.engines.oscillator_momentum_engines import MACD_Engine, RSI_Engine
from indicators.engines.trend_engines import BollingerBands_Engine, EMA_Engine
from indicators.primitives import change, ema, rma, sma, stdev


@pytest.fixture
def series():
    """A deterministic, non-monotonic price series long enough for EMA-200."""
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0, 250.0, size=600)
    return 90_000.0 + np.cumsum(steps)


def first_finite(x):
    finite = np.flatnonzero(np.isfinite(x))
    return int(finite[0]) if finite.size else None


# ── Reference implementations: literal transcriptions of the Pine semantics ──

def ref_sma(x, length):
    out = np.full(len(x), np.nan)
    for i in range(length - 1, len(x)):
        out[i] = sum(x[i - length + 1:i + 1]) / length
    return out


def ref_recursive(x, length, alpha):
    """Pine ta.ema / ta.rma: na until the window fills, SMA seed, then recursion."""
    out = np.full(len(x), np.nan)
    start = next((i for i, v in enumerate(x) if np.isfinite(v)), len(x))
    if start + length > len(x):
        return out
    out[start + length - 1] = sum(x[start:start + length]) / length
    for i in range(start + length, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


# ── Primitives ───────────────────────────────────────────────────────────────

def test_sma_matches_reference(series):
    np.testing.assert_allclose(sma(series, 20), ref_sma(series, 20), rtol=1e-12)


def test_ema_matches_reference(series):
    ref = ref_recursive(series, 26, 2.0 / 27.0)
    np.testing.assert_allclose(ema(series, 26), ref, rtol=1e-12)


def test_rma_matches_reference(series):
    ref = ref_recursive(series, 14, 1.0 / 14.0)
    np.testing.assert_allclose(rma(series, 14), ref, rtol=1e-12)


def test_ema_is_seeded_with_the_simple_average(series):
    """Pine seeds with SMA(length), not with x[0].

    Seeding from the first sample is the common shortcut and stays visibly wrong
    for a long time on a 200-period average.
    """
    out = ema(series, 50)
    assert out[49] == pytest.approx(series[:50].mean(), rel=1e-12)
    assert out[49] != pytest.approx(series[0], rel=1e-6)


def test_stdev_is_population_not_sample(series):
    """ta.stdev defaults to biased=true; ddof=1 would widen every band."""
    out = stdev(series, 20)
    np.testing.assert_allclose(out[19], np.std(series[:20], ddof=0), rtol=1e-12)
    assert out[19] != pytest.approx(np.std(series[:20], ddof=1), rel=1e-9)


@pytest.mark.parametrize("fn,length,expected", [
    (sma, 20, 19),
    (stdev, 20, 19),
    (ema, 26, 25),
    (rma, 14, 13),
])
def test_warmup_is_nan_until_the_window_fills(series, fn, length, expected):
    out = fn(series, length)
    assert np.all(np.isnan(out[:expected]))
    assert np.isfinite(out[expected])
    assert np.all(np.isfinite(out[expected:]))


def test_warmup_counts_from_the_first_finite_value():
    """Pine skips leading na — the window starts at real data, not at bar 0.

    This is what puts the MACD signal line in the right place.
    """
    x = np.concatenate([np.full(10, np.nan), np.arange(1, 51, dtype=np.float64)])
    assert first_finite(ema(x, 5)) == 14      # 10 leading NaN + 5 - 1
    assert first_finite(sma(x, 5)) == 14


def test_change_is_nan_at_the_first_bar(series):
    delta = change(series)
    assert np.isnan(delta[0])
    np.testing.assert_allclose(delta[1:], np.diff(series), rtol=1e-12)


@pytest.mark.parametrize("fn", [sma, stdev, ema, rma])
def test_series_shorter_than_the_window_returns_all_nan(fn):
    """A newly listed symbol has fewer candles than EMA-200 needs."""
    out = fn(np.arange(5, dtype=np.float64), 200)
    assert out.shape == (5,)
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("fn,length", [(sma, 20), (stdev, 20), (ema, 26), (rma, 14)])
def test_output_is_causal(series, fn, length):
    """Changing a bar must not move any earlier output.

    A look-ahead bug is invisible in a value comparison but fatal in backtesting,
    so it gets its own check rather than being assumed.
    """
    k = 400
    mutated = series.copy()
    mutated[k] += 5_000.0
    np.testing.assert_array_equal(fn(series, length)[:k], fn(mutated, length)[:k])


# ── Engines ──────────────────────────────────────────────────────────────────

def run_engine(engine, source):
    out = np.empty((source.shape[0], len(engine.names)), dtype=np.float64, order="F")
    engine.compute(source, out)
    return out


def test_ema_engine_emits_one_column_per_length(series):
    engine = EMA_Engine([9, 26, 50, 100, 200])
    out = run_engine(engine, series)

    assert engine.names == ("ema_9", "ema_26", "ema_50", "ema_100", "ema_200")
    for j, length in enumerate([9, 26, 50, 100, 200]):
        assert first_finite(out[:, j]) == length - 1
        np.testing.assert_allclose(out[:, j], ema(series, length), rtol=1e-12)


def test_macd_signal_warms_up_after_the_macd_itself(series):
    """macd is na until bar 25, so its 9-period EMA cannot start before bar 33."""
    engine = MACD_Engine(12, 26, 9)
    out = run_engine(engine, series)

    assert engine.names == ("macd", "macd_signal", "macd_hist")
    assert first_finite(out[:, 0]) == 25
    assert first_finite(out[:, 1]) == 33
    assert first_finite(out[:, 2]) == 33


def test_macd_histogram_is_line_minus_signal(series):
    out = run_engine(MACD_Engine(12, 26, 9), series)
    valid = np.isfinite(out[:, 2])
    np.testing.assert_allclose(
        out[valid, 2], out[valid, 0] - out[valid, 1], rtol=1e-12
    )


def test_rsi_matches_the_pine_formula(series):
    engine = RSI_Engine(14, 14)
    out = run_engine(engine, series)

    delta = change(series)
    up = rma(np.maximum(delta, 0.0), 14)
    down = rma(-np.minimum(delta, 0.0), 14)
    expected = 100.0 - 100.0 / (1.0 + up / down)

    valid = np.isfinite(out[:, 0])
    np.testing.assert_allclose(out[valid, 0], expected[valid], rtol=1e-12)


def test_rsi_warmup_and_bounds(series):
    out = run_engine(RSI_Engine(14, 14), series)

    assert first_finite(out[:, 0]) == 14     # change is na at bar 0
    assert first_finite(out[:, 1]) == 27     # SMA(14) of a series starting at 14
    rsi = out[np.isfinite(out[:, 0]), 0]
    assert np.all((rsi >= 0.0) & (rsi <= 100.0))


def test_rsi_pins_at_100_when_nothing_falls():
    """down == 0 is Pine's first branch, not a divide-by-zero guard."""
    out = run_engine(RSI_Engine(14, 14), np.arange(1, 61, dtype=np.float64))
    assert out[20, 0] == 100.0


def test_rsi_pins_at_0_when_nothing_rises():
    out = run_engine(RSI_Engine(14, 14), np.arange(60, 0, -1, dtype=np.float64))
    assert out[20, 0] == 0.0


def test_bollinger_bands_bracket_the_basis(series):
    engine = BollingerBands_Engine(20, 2.0)
    out = run_engine(engine, series)

    assert engine.names == ("bb_basis", "bb_upper", "bb_lower")
    assert first_finite(out[:, 0]) == 19

    valid = np.isfinite(out[:, 0])
    basis, upper, lower = out[valid, 0], out[valid, 1], out[valid, 2]
    np.testing.assert_allclose(basis, sma(series, 20)[valid], rtol=1e-12)
    np.testing.assert_allclose(upper - basis, basis - lower, rtol=1e-12)
    np.testing.assert_allclose(
        upper - lower, 4.0 * stdev(series, 20)[valid], rtol=1e-12
    )


def test_bollinger_multiplier_scales_the_width(series):
    narrow = run_engine(BollingerBands_Engine(20, 1.0), series)
    wide = run_engine(BollingerBands_Engine(20, 2.0), series)
    valid = np.isfinite(narrow[:, 1])

    np.testing.assert_allclose(
        wide[valid, 1] - wide[valid, 0],
        2.0 * (narrow[valid, 1] - narrow[valid, 0]),
        rtol=1e-12,
    )
