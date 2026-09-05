"""Path resolution and config loading.

These guard the property that makes the repo runnable on a fresh clone: nothing
resolves to a path outside the project unless the operator asks for it.
"""
import importlib
import os

import pytest

from ingestion import config_fetch_data as cfg


def test_array_path_is_group_root_plus_three_levels():
    path = cfg.build_array_path("Cryptocurrency", "BTC", "BTCUSD")
    assert path == f"{cfg.DB_GROUP_ROOT}/Cryptocurrency/BTC/BTCUSD"


def test_array_path_lives_under_the_data_root():
    path = cfg.build_array_path("Cryptocurrency", "BTC", "BTCUSD")
    assert os.path.commonpath([path, cfg.DATABASE_ROOT_PATH]) == cfg.DATABASE_ROOT_PATH


def test_data_root_defaults_inside_the_repo(monkeypatch):
    """With no env override the data root is <repo>/data — never an absolute
    path baked into the source."""
    monkeypatch.delenv("TSD_DATA_ROOT", raising=False)
    fresh = importlib.reload(cfg)
    try:
        assert fresh.DATABASE_ROOT_PATH == os.path.join(fresh.PROJECT_ROOT, "data")
        assert os.path.isdir(os.path.join(fresh.PROJECT_ROOT, "ingestion"))
    finally:
        # conftest set the env var; restore the module to that state for other tests
        monkeypatch.undo()
        importlib.reload(cfg)


def test_tsd_data_root_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("TSD_DATA_ROOT", str(tmp_path))
    fresh = importlib.reload(cfg)
    try:
        assert fresh.DATABASE_ROOT_PATH == str(tmp_path)
        assert fresh.build_array_path("M", "C", "S").startswith(str(tmp_path))
    finally:
        monkeypatch.undo()
        importlib.reload(cfg)


def test_group_name_is_not_hardcoded_in_the_structure():
    """DATABASE_STRUCTURE keys off DB_GROUP_NAME, so renaming the group in one
    place renames it for the writer too."""
    assert list(cfg.DATABASE_STRUCTURE) == [cfg.DB_GROUP_NAME]


@pytest.mark.parametrize("filename", [
    "all_timeframes.json",
    "continuous_fetch_mode.json",
    "exchange_configs.json",
    "historical_data_config.json",
    "symbols_config.json",
])
def test_config_files_are_found_relative_to_the_package(filename):
    assert os.path.isfile(cfg.get_config_path(filename))


def test_missing_config_fails_loudly():
    with pytest.raises(FileNotFoundError):
        cfg.get_config_path("does_not_exist.json")


def test_loaded_configs_expose_the_keys_the_pipeline_reads():
    assert {"start_date", "fetch_all", "multi_fetch", "multi_write", "write_rate"} <= set(
        cfg.HISTORICAL_DATA_CONFIG
    )
    assert {"continuous", "fetch_interval", "sleep_interval", "rss_restart_mb"} <= set(
        cfg.FETCH_MODE_CONFIG
    )
    assert cfg.SYMBOLS_CONFIG["market"], "at least one market must be configured"

    for name, tf in cfg.TIMEFRAMES.items():
        assert tf["minutes"] > 0, f"timeframe {name} has no duration"


def test_every_exchange_declares_what_the_fetcher_needs():
    for name, exchange in cfg.EXCHANGE_CONFIGS.items():
        assert exchange["api_url"].startswith("https://"), name
        mapping = exchange["format"]["mapping"]
        assert {"timestamp", "open", "high", "low", "close", "volume"} <= set(mapping), name
