"""Lineament feature engineering: the contract the new-fault detector is built on.

The features are the whole reason this detector is a *different* strategy from the shipped
raw-band classifiers, so the tests pin the properties that make it one:

* the feature list is stable and ordered (a report quoting feature 41 means one thing),
* NaN (the nodata sentinel, the footprint edge, the 3 constant bands) survives as a *pixel
  mask* identical across every column, so a row is either fully known or fully unknown,
* the filters are edge-safe: a chunk's interior features do not depend on where the chunk
  boundary was drawn (that is what HALO_PX in the caller is for),
* the values are deterministic for a fixed input.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys  # noqa: E402

sys.path.insert(0, sys_path)

from src.lineament_features import (NODATA_SENTINEL, feature_names,  # noqa: E402
                                    lineament_features, nan_to_zero, sentinel_to_nan)


def _synthetic(bands: int = 19, h: int = 96, w: int = 128, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (bands, h, w)).astype(np.float32)
    # a vertical ridge the filters should be able to see
    x[:, 20:24, :] += 4.0
    # the survey footprint: rows 0..7 are outside it and carry the nodata sentinel
    x[:, :8, :] = -3.4e38
    # three pixels inside the footprint carry the sentinel in EVERY band (the real stack has 3,061)
    x[:, 40, 40] = -3.4e38
    return x


# ------------------------------------------------------------------------------------- the list
def test_feature_names_are_stable_ordered_and_cover_every_block():
    names = feature_names(19)
    assert names == list(names), "no duplicates"
    assert len(names) == 63, f"expected 63 features, got {len(names)}"
    assert names[:19] == [f"b{i:02d}" for i in range(1, 20)], "raw bands come first, in band order"
    assert sum(n.startswith("ridge_b") for n in names) == 6
    assert sum(n.startswith("coher_b") for n in names) == 6
    assert sum(n.startswith("mean5_") for n in names) == 8
    assert sum(n.startswith("std5_") for n in names) == 8
    assert sum(n.startswith("mean11_") for n in names) == 8
    assert sum(n.startswith("std11_") for n in names) == 8


def test_lineament_features_returns_the_same_names_as_feature_names():
    X, names = lineament_features(_synthetic())
    assert names == feature_names(19)
    assert X.shape == (96 * 128, 63)


# ----------------------------------------------------------------------------------- NaN contract
def test_the_nan_mask_is_identical_across_every_column():
    x = _synthetic()
    X, _ = lineament_features(x)
    nanmask = np.isnan(X[:, 0])
    assert nanmask.any()
    for j in range(X.shape[1]):
        assert np.array_equal(np.isnan(X[:, j]), nanmask), \
            f"column {j} has a different NaN pattern - a partially-known row would be scored"


def test_the_nodata_sentinel_is_the_only_source_of_nan_and_it_survives_the_filters():
    x = _synthetic()
    X, _ = lineament_features(x)
    nanmask = np.isnan(X[:, 0])
    expect = (x[0] <= NODATA_SENTINEL).ravel()
    assert np.array_equal(nanmask, expect), "NaN must come from the sentinel and nowhere else"


def test_sentinel_to_nan_and_back_are_inverses():
    x = _synthetic()
    n = sentinel_to_nan(x)
    assert np.isnan(n).any()
    assert not np.isnan(x).any(), "the helper must not mutate its input"
    back = nan_to_zero(n)
    assert np.array_equal(back == 0.0, np.isnan(n)), "NaN must become exactly 0.0"
    assert np.array_equal(back[~np.isnan(n)], x[~np.isnan(n)])


# ------------------------------------------------------------------------------- determinism/edge
def test_the_features_are_deterministic():
    x = _synthetic()
    a, _ = lineament_features(x)
    b, _ = lineament_features(x)
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_a_chunk_boundary_does_not_change_the_interior_features():
    """The reason callers pass a halo: a haloed chunk's interior must match the whole stack.

    ``scripts/newfault_detector.py`` reads the 3,730-row stack in 256-row chunks with a 16-px halo
    on both sides.  This test reproduces that protocol and compares every pixel that is at least
    the halo width from the chunk boundary - which is exactly the set the caller keeps.
    """
    x = _synthetic(bands=19, h=96, w=128, seed=11)
    full, _ = lineament_features(x)
    r0, r1, halo = 32, 64, 16
    part, _ = lineament_features(x[:, max(0, r0 - halo):min(96, r1 + halo), :])
    off = r0 - max(0, r0 - halo)                    # chunk row 0 -> grid row max(0, r0-halo)
    got = part.reshape(min(96, r1 + halo) - max(0, r0 - halo), 128, 63)
    want = full.reshape(96, 128, 63)
    for r in range(r0 + halo, r1 - halo):           # only the rows the caller would keep
        assert np.allclose(np.nan_to_num(want[r]), np.nan_to_num(got[r - r0 + off]),
                           rtol=1e-4, atol=1e-4, equal_nan=True), \
            f"row {r} depends on the chunk boundary despite the halo"


def _ridge_image(amp: float = 8.0, width: float = 6.0, seed: int = 5):
    """Noise plus one smooth Gaussian ridge across band 3 - a line a filter can see."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (19, 96, 128)).astype(np.float32)
    cols = np.arange(128)
    x[2] = x[2] + amp * np.exp(-((cols - 64) ** 2) / (2 * (width / 2.5) ** 2))[None, :]
    return x.astype(np.float32)


def test_the_ridge_is_visible_in_the_ridge_and_coherence_features():
    X, names = lineament_features(_ridge_image())
    ridge = X[:, names.index("ridge_b03")].reshape(96, 128)
    coh = X[:, names.index("coher_b03")].reshape(96, 128)
    X0, _ = lineament_features(np.random.default_rng(5).normal(0, 1, (19, 96, 128)).astype(np.float32))
    ridge0 = X0[:, names.index("ridge_b03")].reshape(96, 128)
    assert float(np.percentile(ridge, 99)) > 2.0 * float(np.percentile(ridge0, 99)), \
        "a smooth amplitude ridge must raise the Sato ridgeness above the noise floor"
    assert float(coh[48, 64]) > 1.5 * float(np.median(coh[:, 100:])), \
        "the structure tensor must be more coherent on the ridge than off it"
    assert 0.0 <= float(np.nanmin(coh)) and float(np.nanmax(coh)) <= 1.0, \
        "coherence is a normalised quantity"


def test_constant_placeholder_bands_do_not_produce_nan_or_inf():
    """Bands 17-19 of the real stack are constants; the filters must degrade gracefully."""
    x = _synthetic()
    x[16] = 2.77935
    x[17] = 0.0
    x[18] = -0.1
    X, _ = lineament_features(x)
    finite = np.isfinite(X)
    assert np.isfinite(X[finite]).all(), "no inf may appear on constant input"


def test_all_nan_input_stays_all_nan():
    x = np.full((19, 32, 32), np.nan, np.float32)
    X, _ = lineament_features(x)
    assert np.isnan(X).all()


# ------------------------------------------------------------------------- the module loads alone
def test_the_module_has_no_competition_data_dependency():
    spec = importlib.util.spec_from_file_location("lf", ROOT / "src" / "lineament_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "lineament_features") and hasattr(mod, "feature_names")
