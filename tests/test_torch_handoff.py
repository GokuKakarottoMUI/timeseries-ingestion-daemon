"""The NumPy → torch hand-off actually shares memory.

"Zero-copy" is easy to claim and easy to lose: one stray ``.contiguous()`` or a
``torch.tensor(...)`` instead of ``torch.from_numpy(...)`` silently doubles the
footprint and nothing fails. These tests pin the property by pointer identity
rather than by inspection.

Skipped cleanly when torch is not installed — it is an optional dependency.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

from get_data.get_data_from_database import BASE_COLS          # noqa: E402
from indicators.feature_pipeline import (                       # noqa: E402
    build_engines,
    compute_block,
    feature_names,
    to_torch,
)

N = 5_000


def numpy_ptr(a: np.ndarray) -> int:
    return a.__array_interface__["data"][0]


@pytest.fixture
def entry():
    rng = np.random.default_rng(3)
    close = 90_000.0 + np.cumsum(rng.normal(0.0, 200.0, size=N))
    block = np.empty((N, len(BASE_COLS)), dtype=np.float64, order="F")
    block[:, 0] = close - 50.0
    block[:, 1] = close + 120.0
    block[:, 2] = close - 120.0
    block[:, 3] = close
    block[:, 4] = rng.uniform(1.0, 50.0, size=N)

    engines = build_engines()
    return {
        "names": feature_names(engines),
        "feat": compute_block(block, block[:, 3], engines),
        "timestamp": np.arange(N, dtype=np.int64) * 60_000,
    }


# ── Memory sharing ───────────────────────────────────────────────────────────

def test_block_tensor_points_at_the_numpy_buffer(entry):
    out = to_torch(entry)
    assert out["block"].data_ptr() == numpy_ptr(entry["feat"])


def test_every_column_tensor_points_at_the_numpy_buffer(entry):
    out = to_torch(entry)
    for j, name in enumerate(entry["names"]):
        assert out[name].data_ptr() == numpy_ptr(entry["feat"][:, j]), name


def test_timestamp_is_shared_and_stays_int64(entry):
    out = to_torch(entry)
    assert out["timestamp"].data_ptr() == numpy_ptr(entry["timestamp"])
    assert out["timestamp"].dtype == torch.int64


def test_writing_through_the_tensor_changes_the_numpy_array(entry):
    """Pointer equality could be coincidence; a write proves the buffer is one."""
    out = to_torch(entry)
    out["close"][0] = 4242.0
    assert entry["feat"][0, entry["names"].index("close")] == 4242.0


# ── Layout: the reason columns are the useful handle ─────────────────────────

def test_column_tensors_are_contiguous(entry):
    """Fortran order was chosen so TileDB could take contiguous columns on write;
    the same layout makes each column directly usable by torch."""
    out = to_torch(entry)
    for name in entry["names"]:
        assert out[name].is_contiguous(), name


def test_whole_block_is_shared_but_not_c_contiguous(entry):
    """Documented caveat: the block hand-off defers a copy, it does not remove
    one. Any op needing C-contiguity will materialise it."""
    out = to_torch(entry)
    assert not out["block"].is_contiguous()
    assert out["block"].stride() == (1, N)


def test_making_the_block_contiguous_copies(entry):
    """The baseline the zero-copy path is measured against."""
    out = to_torch(entry)
    assert out["block"].contiguous().data_ptr() != out["block"].data_ptr()


def test_torch_tensor_constructor_copies(entry):
    """`torch.tensor(feat)` is the easy mistake — it always allocates."""
    assert torch.tensor(entry["feat"]).data_ptr() != numpy_ptr(entry["feat"])


# ── Contract ─────────────────────────────────────────────────────────────────

def test_returns_one_tensor_per_feature_plus_block_and_timestamp(entry):
    out = to_torch(entry)
    assert out["names"] == entry["names"]
    assert out["block"].shape == entry["feat"].shape
    assert set(out) == {"names", "block", "timestamp", *entry["names"]}


def test_values_and_dtype_survive_the_handoff(entry):
    out = to_torch(entry)
    assert out["block"].dtype == torch.float64
    np.testing.assert_array_equal(
        out["close"].numpy(), entry["feat"][:, entry["names"].index("close")]
    )


def test_warmup_nan_is_preserved_not_silently_filled(entry):
    """The NaN policy has to survive the boundary, or a consumer could train on
    values this pipeline never produced."""
    out = to_torch(entry)
    ema200 = out["ema_200"]
    assert torch.isnan(ema200[:199]).all()
    assert torch.isfinite(ema200[199:]).all()


def test_column_name_clashing_with_a_reserved_key_is_rejected(entry):
    """`block`/`timestamp`/`names` share the returned dict with feature names;
    a future indicator called `block` would silently shadow the matrix."""
    entry["names"] = ("block",) + tuple(entry["names"][1:])
    with pytest.raises(ValueError, match="trùng khoá dành riêng"):
        to_torch(entry)
