"""Momentum oscillators.

Ported from ``MACD.pine`` and ``RSI.pine``, numbers only — no plots, no alert
conditions, and no divergence block (``calculateDivergence`` is ``false`` by
default, and pivot-based divergence needs bars to the right of the one being
labelled, which would break causality).
"""
from __future__ import annotations

import numpy as np

from indicators.primitives import change, ema, rma, sma


class MACD_Engine:
    """``MACD.pine`` — MACD line, signal line, histogram.

    Both the oscillator and the signal MA type default to ``"EMA"``.

    The signal line is an EMA *of the MACD*, and the MACD is itself ``na`` while
    the slow EMA warms up. The primitives count their warmup from the first
    finite value, so the signal lands where TradingView puts it rather than 25
    bars early.
    """

    def __init__(self, fast: int, slow: int, signal: int):
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)

    @property
    def names(self) -> tuple[str, ...]:
        return ("macd", "macd_signal", "macd_hist")

    def compute(self, source: np.ndarray, out: np.ndarray) -> None:
        macd = ema(source, self.fast) - ema(source, self.slow)
        signal = ema(macd, self.signal)

        out[:, 0] = macd
        out[:, 1] = signal
        np.subtract(macd, signal, out=out[:, 2])   # histogram


class RSI_Engine:
    """``RSI.pine`` — Wilder RSI plus the script's default smoothing MA.

    ``maTypeInput`` defaults to ``"SMA"`` with length 14, so the stock indicator
    really does draw an SMA of the RSI; it is emitted as ``rsi_ma_<ma_length>``.

    The two guard branches are Pine's, not a division-by-zero patch: a window
    with no downward movement pins the RSI at 100, and one with no upward
    movement pins it at 0. ``down == 0`` is checked first, matching the order of
    the ternary in the script.
    """

    def __init__(self, length: int, ma_length: int):
        self.length = int(length)
        self.ma_length = int(ma_length)

    @property
    def names(self) -> tuple[str, ...]:
        return ("rsi_%d" % self.length, "rsi_ma_%d" % self.ma_length)

    def compute(self, source: np.ndarray, out: np.ndarray) -> None:
        delta = change(source)
        up = rma(np.maximum(delta, 0.0), self.length)
        down = rma(-np.minimum(delta, 0.0), self.length)

        # NaN flows through untouched: the comparisons below are False for NaN,
        # so warmup bars fall to the arithmetic branch and stay NaN.
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = 100.0 - 100.0 / (1.0 + up / down)
        rsi = np.where(down == 0.0, 100.0, np.where(up == 0.0, 0.0, rsi))

        out[:, 0] = rsi
        out[:, 1] = sma(rsi, self.ma_length)
