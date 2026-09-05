"""Response parsing — the config-driven adapter layer.

Two exchanges are shaped very differently (Binance returns a list of arrays with
string numbers; Bitstamp returns a nested object with second-resolution
timestamps) and both must come out as the same ``(N, 6)`` float64 array.
"""
import orjson
import numpy as np
import pytest

from ingestion.exchange_utils import ExchangeFormatter

# Real-shaped Binance kline rows: [openTime, o, h, l, c, v, closeTime, ...].
# Numbers arrive as strings — the parser must coerce them in C, not per row.
BINANCE_PAYLOAD = [
    [1735689600000, "93500.10", "94000.50", "93000.00", "93800.25", "1234.5678",
     1735693199999, "0", 0, "0", "0", "0"],
    [1735693200000, "93800.25", "95000.00", "93700.00", "94900.75", "2345.6789",
     1735696799999, "0", 0, "0", "0", "0"],
]

# Bitstamp: nested under data.ohlc, timestamps in SECONDS as strings.
BITSTAMP_PAYLOAD = {
    "data": {
        "pair": "BTC/USD",
        "ohlc": [
            {"timestamp": "1735689600", "open": "93500.10", "high": "94000.50",
             "low": "93000.00", "close": "93800.25", "volume": "1234.5678"},
            {"timestamp": "1735693200", "open": "93800.25", "high": "95000.00",
             "low": "93700.00", "close": "94900.75", "volume": "2345.6789"},
        ],
    }
}


def test_binance_array_response_maps_columns_by_index():
    out = ExchangeFormatter.parse_response(orjson.dumps(BINANCE_PAYLOAD), "binance")

    assert out.shape == (2, 6)
    assert out.dtype == np.float64
    # [timestamp, open, high, low, close, volume] — note col 6 (closeTime) is dropped
    np.testing.assert_allclose(
        out[0], [1735689600000, 93500.10, 94000.50, 93000.00, 93800.25, 1234.5678]
    )
    np.testing.assert_allclose(
        out[1], [1735693200000, 93800.25, 95000.00, 93700.00, 94900.75, 2345.6789]
    )


def test_bitstamp_object_response_maps_columns_by_key():
    out = ExchangeFormatter.parse_response(orjson.dumps(BITSTAMP_PAYLOAD), "bitstamp")

    assert out.shape == (2, 6)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out[:, 1], [93500.10, 93800.25])
    np.testing.assert_allclose(out[:, 5], [1234.5678, 2345.6789])


def test_bitstamp_seconds_are_scaled_to_milliseconds():
    """Bitstamp declares timestamp_multiplier=1000; both exchanges must end up
    on the same ms-epoch scale so they can share one TileDB dimension."""
    binance = ExchangeFormatter.parse_response(orjson.dumps(BINANCE_PAYLOAD), "binance")
    bitstamp = ExchangeFormatter.parse_response(orjson.dumps(BITSTAMP_PAYLOAD), "bitstamp")

    np.testing.assert_array_equal(binance[:, 0], bitstamp[:, 0])
    assert bitstamp[0, 0] == 1735689600000


def test_ohlc_relationships_survive_parsing():
    out = ExchangeFormatter.parse_response(orjson.dumps(BINANCE_PAYLOAD), "binance")
    high, low = out[:, 2], out[:, 3]
    assert np.all(high >= out[:, 1]) and np.all(high >= out[:, 4])
    assert np.all(low <= out[:, 1]) and np.all(low <= out[:, 4])


@pytest.mark.parametrize("raw,exchange", [
    (b"[]", "binance"),                                    # empty list
    (b"{}", "bitstamp"),                                   # missing data_path
    (b'{"code":-1121,"msg":"Invalid symbol."}', "binance"),  # API error object
    (b'{"data":{"ohlc":[]}}', "bitstamp"),                 # empty ohlc
    (b"not json at all", "binance"),                       # malformed body
])
def test_bad_payloads_return_empty_instead_of_raising(raw, exchange):
    """A fetch loop covering thousands of requests must not die on one bad
    response — the parser degrades to an empty batch."""
    out = ExchangeFormatter.parse_response(raw, exchange)
    assert out.shape == (0, 6)
    assert out.dtype == np.float64


def test_object_response_skips_items_missing_required_fields():
    payload = {"data": {"ohlc": [
        {"timestamp": "1735689600", "open": "1", "high": "2", "low": "0.5", "close": "1.5",
         "volume": "10"},
        {"timestamp": "1735693200", "open": "1"},                       # truncated → dropped
        {"timestamp": "1735696800", "open": "2", "high": "3", "low": "1", "close": "2.5"},
    ]}}
    out = ExchangeFormatter.parse_response(orjson.dumps(payload), "bitstamp")

    assert out.shape == (2, 6)
    assert out[1, 5] == 0.0, "missing volume defaults to 0, it does not drop the candle"


@pytest.mark.parametrize("timeframe,expected", [
    ("1h", "3600"),
    ("15m", "900"),
    ("1d", "86400"),
])
def test_timeframe_is_rendered_in_the_unit_the_exchange_wants(timeframe, expected):
    """Bitstamp declares timeframe_format=seconds."""
    assert ExchangeFormatter.format_timeframe(timeframe, "bitstamp") == expected


def test_unknown_timeframe_passes_through_unchanged():
    assert ExchangeFormatter.format_timeframe("7h", "bitstamp") == "7h"
