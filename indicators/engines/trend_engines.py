"""Moving averages and volatility bands.

Ported from ``EMA.pine`` and ``Bollinger_Bands.pine``. Everything the scripts do
for display — plots, colours, fills, offsets — is dropped; only the numbers a
model can consume survive.

Each engine writes straight into a caller-provided column slice, so the pipeline
allocates the feature block once and no intermediate array is ever materialised.
"""
from __future__ import annotations

import numpy as np

from indicators.primitives import ema, sma, stdev


class EMA_Engine:
    """``EMA.pine`` — one exponential moving average per configured length.

    The script exposes a single ``Length`` input and plots one curve; a model
    wants the whole fan at once, so the engine takes the set of lengths and
    fills them in one pass.

    Its smoothing MA block is left out: ``maTypeInput`` defaults to ``"None"``,
    so the stock indicator emits nothing beyond the EMA itself.
    """

    def __init__(self, lengths):
        self.lengths = [int(v) for v in lengths]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"ema_{length}" for length in self.lengths)

    def compute(self, source: np.ndarray, out: np.ndarray) -> None:
        # One lfilter pass per length: each length has its own alpha, so the
        # recursions cannot be folded into a single call. The loop runs over the
        # handful of lengths, never over candles.
        for j, length in enumerate(self.lengths):
            out[:, j] = ema(source, length)


class BollingerBands_Engine:
    """``Bollinger_Bands.pine`` — SMA basis with population-stdev bands.

    Defaults follow the script: length 20, basis MA type ``"SMA"``, multiplier
    2.0. ``ta.stdev`` is biased (population); see ``primitives.stdev``.
    """

    def __init__(self, length: int, mult: float):
        self.length = int(length)
        self.mult = float(mult)

    @property
    def names(self) -> tuple[str, ...]:
        return ("bb_basis", "bb_upper", "bb_lower")

    def compute(self, source: np.ndarray, out: np.ndarray) -> None:
        basis = sma(source, self.length)
        dev = stdev(source, self.length)
        dev *= self.mult                      # in place: dev is already ours

        out[:, 0] = basis
        np.add(basis, dev, out=out[:, 1])     # upper
        np.subtract(basis, dev, out=out[:, 2])  # lower
