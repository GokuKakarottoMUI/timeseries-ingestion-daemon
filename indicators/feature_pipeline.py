"""Turn stored candles into a feature matrix.

Reads through the existing query layer (``get_data``) rather than talking to
TileDB directly, so the reader's locking, retry and block-assembly behaviour is
shared rather than duplicated.

Output per (symbol, timeframe) is a single Fortran-ordered float64 block: the
OHLCV columns exactly as stored, with the indicator columns appended beside
them. One allocation, one column layout, no intermediate matrices.

Warmup stays NaN. Nothing is forward-filled and no sentinel is substituted, so
the caller decides where its usable history begins instead of inheriting a
choice made here.
"""
from __future__ import annotations

import numpy as np
import picologging as logging

from get_data.get_data_from_database import BASE_COLS, GetDataFromDatabase
from indicators.config.config_indicators import COMPUTE_CONFIG, INDICATORS_CONFIG
from indicators.engines.oscillator_momentum_engines import MACD_Engine, RSI_Engine
from indicators.engines.trend_engines import BollingerBands_Engine, EMA_Engine

logger = logging.getLogger("feature_pipeline")


def build_engines(config: dict | None = None) -> list:
    """Instantiate every engine from configuration, in output-column order."""
    cfg = INDICATORS_CONFIG if config is None else config
    return [
        EMA_Engine(cfg["ema"]["lengths"]),
        MACD_Engine(cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"]),
        RSI_Engine(cfg["rsi"]["length"], cfg["rsi"]["ma_length"]),
        BollingerBands_Engine(cfg["bbands"]["length"], cfg["bbands"]["mult"]),
    ]


def feature_names(engines: list) -> tuple[str, ...]:
    """Column names of the assembled block: the stored columns, then indicators."""
    names: list[str] = list(BASE_COLS)
    for engine in engines:
        names.extend(engine.names)
    return tuple(names)


def compute_block(block: np.ndarray, source: np.ndarray, engines: list,
                  dtype=np.float64) -> np.ndarray:
    """Assemble one ``(N, len(BASE_COLS) + n_indicators)`` feature block.

    Args:
        block:  ``(N, len(BASE_COLS))`` float64 OHLCV, as returned by get_data.
        source: ``(N,)`` price series the indicators read (a column view of
                ``block`` — the scripts all take ``close``).
        engines: engines in column order, from ``build_engines``.
        dtype:  output dtype; float64 keeps the pipeline lossless.

    Returns:
        Fortran-ordered array whose first ``len(BASE_COLS)`` columns are the
        input block unchanged.
    """
    n_base = len(BASE_COLS)
    n_ind = sum(len(engine.names) for engine in engines)

    # Single allocation. Fortran order so every column — base and indicator
    # alike — is contiguous for whatever consumes it next.
    feat = np.empty((block.shape[0], n_base + n_ind), dtype=dtype, order="F")
    feat[:, :n_base] = block

    col = n_base
    for engine in engines:
        width = len(engine.names)
        engine.compute(source, feat[:, col:col + width])
        col += width
    return feat


def run(query=None) -> dict:
    """Compute features for every active symbol and timeframe.

    Args:
        query: object exposing ``query_data_for_training()``; defaults to a
            fresh ``GetDataFromDatabase``. Injectable so tests can drive the
            pipeline without standing up a reader.

    Returns:
        ``{(market, symbol_category, symbol, timeframe): {"names", "feat", "timestamp"}}``
        where ``feat`` is ``(T, len(names))`` and ``timestamp`` is ``int64[T]``,
        row-aligned with ``feat``.
    """
    reader = GetDataFromDatabase() if query is None else query
    engines = build_engines()
    names = feature_names(engines)
    dtype = np.dtype(COMPUTE_CONFIG.get("output_dtype", "float64"))

    data = reader.query_data_for_training()
    result: dict[tuple, dict] = {}

    for market, market_data in data.items():
        for category, category_data in market_data.items():
            for symbol, timeframes in category_data.items():
                for tf_name, tf_block in timeframes.items():
                    feat = compute_block(
                        tf_block["block"], tf_block["close"], engines, dtype
                    )
                    result[(market, category, symbol, tf_name)] = {
                        "names": names,
                        "feat": feat,
                        "timestamp": tf_block["timestamp"],
                    }
                    logger.info(
                        f"{market}/{category}/{symbol} @ {tf_name}: "
                        f"feat {feat.shape} {feat.dtype}"
                    )

    if not result:
        logger.warning("Không có block nào để tính feature. Fetch dữ liệu trước.")
    return result
