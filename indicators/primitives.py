"""Pine Script moving-average and dispersion primitives, vectorized.

These reproduce TradingView's ``ta.*`` semantics exactly, because an indicator
that is 99% right is worth nothing: it disagrees with the chart the strategy was
designed on, silently.

Two details carry most of the risk:

1. **Warmup is ``na``, not a partial value.** ``ta.sma(x, L)`` yields ``na`` for
   the first ``L-1`` bars; ``ta.ema``/``ta.rma`` do the same and then *seed* with
   the simple average of the first ``L`` values. Seeding from ``x[0]`` instead —
   the common ``adjust=False`` shortcut — produces a curve that never quite
   matches TradingView and converges only slowly.
2. **Leading ``na`` shifts the warmup window.** Pine starts counting from the
   first non-``na`` value, not from bar 0. This matters for chained indicators:
   the MACD signal line is an EMA of the MACD, which is itself ``na`` for its
   first 25 bars.

NaN is deliberately preserved throughout — there is no forward-fill and no
sentinel. A caller that needs a dense matrix decides for itself where to cut.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter


def _first_finite(x: np.ndarray) -> int:
    """Index of the first finite value, or ``x.size`` when there is none."""
    finite = np.flatnonzero(np.isfinite(x))
    return int(finite[0]) if finite.size else int(x.size)


def _empty_like(x: np.ndarray) -> np.ndarray:
    """All-NaN float64 output of the same length — the 'not enough data' answer."""
    return np.full(x.shape, np.nan, dtype=np.float64)


def sma(x: np.ndarray, length: int) -> np.ndarray:
    """``ta.sma`` — rolling arithmetic mean, ``na`` until the window is full.

    Uses a sliding-window *view* (zero-copy, C-level reduction) rather than the
    usual ``cumsum`` difference. On a long BTC series the running sum reaches
    ~5e10 while the quantity of interest is ~1e5, so the subtraction throws away
    significant digits; the windows here are small enough (14, 20) that the
    exact form costs nothing worth saving.
    """
    n = x.shape[0]
    out = _empty_like(x)
    f = _first_finite(x)
    if length < 1 or f + length > n:
        return out
    out[f + length - 1:] = sliding_window_view(x[f:], length).mean(axis=-1)
    return out


def stdev(x: np.ndarray, length: int) -> np.ndarray:
    """``ta.stdev`` — rolling standard deviation, **population** (``ddof=0``).

    Pine's ``ta.stdev`` defaults to ``biased=true``. Using the sample form
    (``ddof=1``) instead shifts every Bollinger band outward by a constant
    factor — a systematic error that never raises and is easy to miss on a chart.
    """
    n = x.shape[0]
    out = _empty_like(x)
    f = _first_finite(x)
    if length < 1 or f + length > n:
        return out
    out[f + length - 1:] = sliding_window_view(x[f:], length).std(axis=-1)
    return out


def _recursive_ma(x: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """Shared body of ``ema`` and ``rma`` — they differ only in ``alpha``.

    The recursion ``y[i] = alpha*x[i] + (1-alpha)*y[i-1]`` is an IIR filter, so
    it goes to ``scipy.signal.lfilter`` (C/Fortran core) rather than a Python
    loop. The Pine seed is expressed as the filter's initial condition ``zi``,
    which is what makes the first recursive step land on the right value.
    """
    n = x.shape[0]
    out = _empty_like(x)
    f = _first_finite(x)
    if length < 1 or f + length > n:
        return out

    seed = float(x[f:f + length].mean())
    seed_idx = f + length - 1
    out[seed_idx] = seed

    tail = x[seed_idx + 1:]
    if tail.size:
        # zi holds (1-alpha)*y[seed_idx] so that the first output of the filter
        # is alpha*x[seed_idx+1] + (1-alpha)*seed, exactly as Pine continues.
        filtered, _ = lfilter(
            np.array([alpha], dtype=np.float64),
            np.array([1.0, -(1.0 - alpha)], dtype=np.float64),
            tail,
            zi=np.array([(1.0 - alpha) * seed], dtype=np.float64),
        )
        out[seed_idx + 1:] = filtered
    return out


def ema(x: np.ndarray, length: int) -> np.ndarray:
    """``ta.ema`` — exponential MA, ``alpha = 2/(length+1)``, SMA-seeded."""
    return _recursive_ma(x, length, 2.0 / (length + 1.0))


def rma(x: np.ndarray, length: int) -> np.ndarray:
    """``ta.rma`` — Wilder's smoothing, ``alpha = 1/length``, SMA-seeded."""
    return _recursive_ma(x, length, 1.0 / length)


def change(x: np.ndarray) -> np.ndarray:
    """``ta.change`` — first difference, ``na`` at bar 0."""
    out = _empty_like(x)
    if x.shape[0] > 1:
        np.subtract(x[1:], x[:-1], out=out[1:])
    return out
