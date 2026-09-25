"""Lineament (ridge/valley) feature engineering for the competition's 19-band stack.

WHY A MODULE OF ITS OWN
-----------------------
Faults in geophysical rasters are *lineaments*: narrow, elongated anomalies in a
gradient field.  A per-pixel classifier (see ``scripts/baseline_submission.py``)
sees each band's value in isolation and therefore cannot distinguish "this pixel
sits on a 300 m-wide linear magnetic break" from "this pixel sits in a broad
blobby magnetic high".  The filters here add the missing geometry explicitly:

* ``sato`` ridgeness (Sato et al. 1998, the Frangi-family tubeness filter) at
  scales 1-3 px - a *multi-scale* line detector, because fault expressions in
  the stack range from one cell (a sharp magnetic break) to several cells (a
  broad gravity lineament);
* the structure-tensor *coherence* ``(l1 - l2) / (l1 + l2)`` at sigma 1.5 px -
  the orientation-independent line-ness of the local gradient field, which is
  scale-tuned differently from the Hessian filters and therefore not redundant;
* windowed mean / standard deviation at 5 px and 11 px - local context, so the
  classifier can tell "an isolated spike" from "a sustained step".

Only bands that carry an edge/line signal get the expensive treatment; the
remaining bands still contribute their raw value and their windowed statistics.
The band indices are 1-based, matching the ``description`` tags rasterio reads
out of ``training_features.tif`` (verified 2026-09-25 against the official
stack, sha256 ``4371c82e...``):

    3  Total magnetic intensity horizontal gradient
    6  Tilt angle or total curvature
    9  Total magnetic intensity vertical gradient
    11 Isostatic gravity anomaly vertical gradient
    18 Isostatic gravity anomaly horizontal gradient
    19 Detrended elevation slope
    12 Detrended elevation
    13 Isostatic gravity anomaly

NaN CONTRACT
------------
The official stack encodes "no measurement" as ``-3.4e38``.  Filters smear
whatever they are given, so every band is filled with 0 before filtering and the
result is set back to NaN wherever the input was a sentinel - the classifier
(``HistGradientBoostingClassifier``) consumes NaN natively, and leaving the
sentinels as NaN keeps the feature matrix honest about where the data ends
(3,061 such pixels sit inside the survey footprint - the same holes
``scripts/sanitize_submission.py`` has to fill in a submission).

Nothing here is fitted: the same pixels always produce the same features, so the
module is deterministic and testable on a synthetic array.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SATO_BANDS", "COHERENCE_BANDS", "WINDOW_BANDS", "FEATURE_NAMES",
           "sentinel_to_nan", "nan_to_zero", "lineament_features", "feature_names"]

# 1-based band numbers of training_features.tif that carry a line signal.
SATO_BANDS: tuple[int, ...] = (3, 6, 9, 11, 18, 19)
COHERENCE_BANDS: tuple[int, ...] = (3, 6, 9, 11, 18, 19)
WINDOW_BANDS: tuple[int, ...] = (1, 2, 3, 6, 9, 12, 13, 19)
SIGMAS: tuple[float, ...] = (1.0, 2.0, 3.0)
COHERENCE_SIGMA = 1.5
WINDOW_SIZES: tuple[int, ...] = (5, 11)
NODATA_SENTINEL = -1e30


def sentinel_to_nan(arr: np.ndarray) -> np.ndarray:
    """``-3.4e38`` -> NaN, everything else untouched (float32 copy)."""
    out = np.asarray(arr, dtype=np.float32).copy()
    out[out < NODATA_SENTINEL] = np.nan
    return out


def nan_to_zero(arr: np.ndarray) -> np.ndarray:
    """NaN -> 0.0 so a filter does not propagate the hole across the raster."""
    out = np.array(arr, dtype=np.float32, copy=True)
    out[~np.isfinite(out)] = 0.0
    return out


def feature_names(n_bands: int) -> list[str]:
    """Ordered feature names for :func:`lineament_features` with ``n_bands`` raw bands."""
    names = [f"b{i:02d}" for i in range(1, n_bands + 1)]
    names += [f"ridge_b{i:02d}" for i in SATO_BANDS if i <= n_bands]
    names += [f"coher_b{i:02d}" for i in COHERENCE_BANDS if i <= n_bands]
    for i in WINDOW_BANDS:
        if i > n_bands:
            continue
        for w in WINDOW_SIZES:
            names.append(f"mean{w}_b{i:02d}")
            names.append(f"std{w}_b{i:02d}")
    return names


def _sato_ridgeness(band: np.ndarray, sigmas=SIGMAS) -> np.ndarray:
    """Multi-scale ridge-ness in [0, 1]; NaN wherever the band has no measurement."""
    from skimage.filters import sato
    filled = nan_to_zero(band)
    with np.errstate(invalid="ignore", divide="ignore"):
        resp = sato(filled, sigmas=list(sigmas), black_ridges=False)
    resp = np.asarray(resp, dtype=np.float32)
    resp[~np.isfinite(resp)] = 0.0
    resp[~np.isfinite(band)] = np.nan
    return np.clip(resp, 0.0, None).astype(np.float32)


def _coherence(band: np.ndarray, sigma: float = COHERENCE_SIGMA) -> np.ndarray:
    """Structure-tensor coherence (l1 - l2) / (l1 + l2) in [0, 1]."""
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    filled = nan_to_zero(band)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = structure_tensor(filled, sigma=sigma)
        l1, l2 = structure_tensor_eigenvalues(np.stack(a))
    denom = np.abs(l1) + np.abs(l2)
    coh = np.where(denom > 0, (np.abs(l1) - np.abs(l2)) / np.where(denom > 0, denom, 1.0), 0.0)
    coh = np.asarray(coh, dtype=np.float32)
    coh[~np.isfinite(coh)] = 0.0
    coh[~np.isfinite(band)] = np.nan
    return np.clip(coh, 0.0, 1.0).astype(np.float32)


def _window_stats(band: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and standard deviation over a ``size`` x ``size`` window."""
    from scipy.ndimage import uniform_filter
    filled = nan_to_zero(band)
    finite = np.isfinite(band)
    weight = finite.astype(np.float32)
    n = uniform_filter(weight, size=size, mode="nearest")
    safe_n = np.maximum(n, 1e-6)
    mean = uniform_filter(np.where(finite, filled, 0.0), size=size, mode="nearest") / safe_n
    mean = np.where(n > 0.5, mean, 0.0)
    sq = uniform_filter(np.where(finite, filled * filled, 0.0), size=size, mode="nearest") / safe_n
    var = np.maximum(sq - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(n > 0.5, std, 0.0)
    mean[~finite] = np.nan
    std[~finite] = np.nan
    return mean.astype(np.float32), std.astype(np.float32)


def lineament_features(bands: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build the model matrix from a ``(n_bands, H, W)`` stack.

    Returns ``(X, names)`` with ``X`` shaped ``(H*W, n_features)`` in C order and
    NaN wherever the pixel has no measurement.  The row order is
    ``row * W + col`` so a caller can reshape the prediction back onto the grid.
    """
    bands = np.asarray(bands)
    if bands.ndim != 3:
        raise ValueError(f"expected (n_bands, H, W), got {bands.shape}")
    n_bands, H, W = bands.shape
    cols: list[np.ndarray] = []
    for i in range(n_bands):
        cols.append(sentinel_to_nan(bands[i]).ravel())
    for b in SATO_BANDS:
        if b <= n_bands:
            cols.append(_sato_ridgeness(sentinel_to_nan(bands[b - 1])).ravel())
    for b in COHERENCE_BANDS:
        if b <= n_bands:
            cols.append(_coherence(sentinel_to_nan(bands[b - 1])).ravel())
    for b in WINDOW_BANDS:
        if b > n_bands:
            continue
        band = sentinel_to_nan(bands[b - 1])
        for size in WINDOW_SIZES:
            mean, std = _window_stats(band, size)
            cols.append(mean.ravel())
            cols.append(std.ravel())
    X = np.empty((H * W, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, j] = c
    return X, feature_names(n_bands)
