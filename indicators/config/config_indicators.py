import os
import yaml
from yaml import CSafeLoader
from typing import Any

# ─────────────────────────────────────────────
# PATH HELPER — config nằm cùng thư mục này (indicators/config/)
# ─────────────────────────────────────────────
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

def get_indicators_config_path(filename: str) -> str:
    path = os.path.join(_CONFIG_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return path

# ─────────────────────────────────────────────
# GENERIC YAML LOADER (CSafeLoader = libyaml C core)
# ─────────────────────────────────────────────
def _load_yaml(filename: str) -> Any:
    """Load YAML bằng CSafeLoader (libyaml C core). Chỉ gọi 1 lần khi load module."""
    with open(get_indicators_config_path(filename), 'rb') as f:
        return yaml.load(f, Loader=CSafeLoader)

# ─────────────────────────────────────────────
# INDICATORS CONFIG — tham số từng indicator (lengths, mult, source)
# COMPUTE CONFIG    — dtype output
# ─────────────────────────────────────────────
INDICATORS_CONFIG = _load_yaml('indicators_config.yaml')
COMPUTE_CONFIG    = _load_yaml('compute_config.yaml')
