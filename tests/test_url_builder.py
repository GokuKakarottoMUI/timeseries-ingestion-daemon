"""Request building — URL templates, unit conversion and symbol variants.

Adding an exchange should be a config change, so these tests drive the builder
purely through what the JSON declares.
"""
import ingestion.exchange_utils as exchange_utils
from ingestion.exchange_utils import ExchangeURLBuilder, _VARIANT_LOOKUP

# 2025-01-01T00:00:00Z .. 2025-01-02T00:00:00Z in ms
START_MS = 1735689600000
END_MS = 1735776000000


def test_symbol_placeholder_in_the_url_is_substituted():
    url, _ = ExchangeURLBuilder.get_request_params("bitstamp", "BTCUSD", "1h", START_MS, END_MS)
    assert "{symbol}" not in url
    assert url.startswith("https://www.bitstamp.net/api/v2/ohlc/")


def test_params_come_from_config_and_use_the_exchange_unit():
    """Bitstamp declares timestamp_format=seconds, so the ms-epoch the pipeline
    works in must be divided down at the request boundary — and only there."""
    _, params = ExchangeURLBuilder.get_request_params(
        "bitstamp", "BTCUSD", "1h", START_MS, END_MS
    )
    assert params["step"] == "3600"          # 1h rendered in seconds
    assert params["start"] == str(START_MS // 1000)
    assert int(params["end"]) <= END_MS // 1000
    assert params["limit"] == 1000


def test_end_is_clamped_to_one_limit_window():
    """Requesting a year must not ask the exchange for a year: the window is
    capped at limit × timeframe so a response can never overflow `limit`."""
    far_end = START_MS + 365 * 24 * 3600 * 1000
    _, params = ExchangeURLBuilder.get_request_params(
        "bitstamp", "BTCUSD", "1h", START_MS, far_end, limit=1000
    )
    window_s = int(params["end"]) - int(params["start"])
    assert window_s == 1000 * 3600


def test_custom_limit_shrinks_the_window():
    _, params = ExchangeURLBuilder.get_request_params(
        "bitstamp", "BTCUSD", "1h", START_MS, END_MS, limit=10
    )
    assert params["limit"] == 10
    assert int(params["end"]) - int(params["start"]) == 10 * 3600


def test_inactive_exchange_yields_no_request(monkeypatch):
    """A disabled exchange must be refused, not silently requested.

    The active flag is pinned here rather than read from the shipped JSON: which
    exchanges are enabled is runtime state an operator toggles, and a test that
    depends on it reports on a config file instead of on the builder.
    """
    disabled = dict(exchange_utils.EXCHANGE_CONFIGS["bitstamp"], active=False)
    monkeypatch.setitem(exchange_utils.EXCHANGE_CONFIGS, "bitstamp", disabled)

    url, params = ExchangeURLBuilder.get_request_params(
        "bitstamp", "BTCUSD", "1h", START_MS, END_MS
    )
    assert (url, params) == ("", {})


def test_unknown_exchange_yields_no_request():
    url, params = ExchangeURLBuilder.get_request_params(
        "nope", "BTCUSD", "1h", START_MS, END_MS
    )
    assert (url, params) == ("", {})


def test_variant_lookup_is_built_from_active_symbols_only():
    assert "BTCUSD" in _VARIANT_LOOKUP
    variants = _VARIANT_LOOKUP["BTCUSD"]
    assert variants, "an active symbol must expose at least one ticker variant"
    # Exchanges disagree on case; both spellings are offered without duplicates
    assert "BTCUSD" in variants and "btcusd" in variants
    assert len(variants) == len(set(variants))


def test_unknown_symbol_falls_back_to_the_name_given():
    assert ExchangeURLBuilder._find_exchange_symbol("bitstamp", "NOSUCHPAIR") == "NOSUCHPAIR"


def test_successful_variant_is_cached_for_reuse():
    ExchangeURLBuilder.update_successful_symbol("bitstamp", "BTCUSD", "btcusd")
    try:
        assert ExchangeURLBuilder._find_exchange_symbol("bitstamp", "BTCUSD") == "btcusd"
    finally:
        ExchangeURLBuilder._successful_symbols.pop("bitstamp_BTCUSD", None)
