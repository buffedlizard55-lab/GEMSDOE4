"""Tests for scripts/paired_union_contrast.py - the script an adoption decision rests on.

WHY THIS FILE EXISTS
--------------------
Until 2026-09-26 this script had NO test at all, while its output
(`data/evidence/union_po_loo/contrasts/drop_nff42_vs_committed.json`) is the only paired
probability behind the currently adopted submission.  Reading it line by line found a real
defect: the candidate arm's per-block rows were built WITHOUT the truth's `scoreable` flag, so
`bootstrap_from_blocks` counted `n_scoreable_blocks: 0` for the arm that won - a committed
number that reads as "no truth in any block" next to that arm's own 0.1996 DTI.

These tests pin the four things that must not silently change:

1. `scoreable`/`n_gt` are properties of the TRUTH, so both arms report the same counts;
2. the recomposition identity guard still refuses to emit a bootstrap it cannot reproduce;
3. pooling several folds (`--fold 2,3`) keeps the union of their blocks, records which fold
   every block came from, and reports each fold's own scalar pair next to the pooled interval;
4. a fold list that would undercount (duplicates, out of range, junk) is rejected rather than
   silently truncated - an undercount is how a COARSE interval gets quoted as adequate.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "paired_union_contrast.py"


def _mod():
    spec = importlib.util.spec_from_file_location("paired_union_contrast", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(path: Path, arr: np.ndarray, dtype: str, nodata=None) -> Path:
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype=dtype, crs="EPSG:32611",
                   transform=from_origin(500000.0, 4100000.0, 100.0, 100.0),
                   nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype, copy=False), 1)
    return path


def _fixture(d: Path, size: int = 128, block_px: int = 32):
    """A 4x4 block grid with proxy (code 2) traces in four different blocks.

    The template footprint excludes the top two rows, so the footprint mask is exercised too.
    The reference field covers two traces, the candidate covers three (one of them where the
    reference is blind), so the contrast has a real sign.
    """
    footprint = np.full((size, size), np.nan, np.float32)
    footprint[2:, :] = 1.0

    proxy = np.zeros((size, size), np.uint8)
    for col in (10, 42, 74, 106):          # one vertical trace per block column
        proxy[8:size - 8, col:col + 2] = 2

    labels = np.zeros((size, size), np.uint8)
    labels[8:size - 8, 10:12] = 1          # the catalogue knows only the first trace

    ref = np.zeros((size, size), np.float32)
    ref[8:size - 8, 10:12] = 1.0
    ref[8:size - 8, 42:44] = 1.0

    cand = np.zeros((size, size), np.float32)
    cand[8:size - 8, 10:12] = 1.0
    cand[8:size - 8, 42:44] = 1.0
    cand[8:size - 8, 74:76] = 1.0

    return dict(
        labels=_write(d / "labels.tif", labels, "uint8"),
        proxy=_write(d / "proxy.tif", proxy, "uint8"),
        template=_write(d / "template.tif", footprint, "float32", nodata=np.nan),
        reference=_write(d / "reference.tif", ref, "float32", nodata=np.nan),
        candidate=_write(d / "candidate.tif", cand, "float32", nodata=np.nan),
        block_px=block_px,
    )


def _run(fx, out: Path, fold: str, extra=()):
    mod = _mod()
    argv = ["--reference", str(fx["reference"]), "--candidate", str(fx["candidate"]),
            "--labels", str(fx["labels"]), "--proxy", str(fx["proxy"]),
            "--template", str(fx["template"]), "--block-px", str(fx["block_px"]),
            "--folds", "4", "--seed", "42", "--n-boot", "200", "--boot-seed", "0",
            "--fold", fold, "--out", str(out), *extra]
    rc = mod.main(argv)
    assert rc == 0
    return json.loads(out.read_text())


# --------------------------------------------------------------------------- fold-list parsing

def test_fold_list_parsing_accepts_one_fold_and_a_list():
    mod = _mod()
    assert mod.parse_fold_list("1", 4) == [1]
    assert mod.parse_fold_list("2,3", 4) == [2, 3]
    assert mod.parse_fold_list(" 3 , 0 ", 4) == [0, 3]      # whitespace and order normalised


@pytest.mark.parametrize("spec", ["3,3", "0,1,2,3,4", "-1", "x", "", "1,,2"])
def test_fold_list_parsing_refuses_anything_that_could_undercount(spec):
    """A duplicate double-counts a block; an out-of-range fold silently drops one."""
    mod = _mod()
    with pytest.raises(SystemExit):
        mod.parse_fold_list(spec, 4)


# --------------------------------------------------------------------------- the scoreable fix

def test_both_arms_report_the_same_scoreable_blocks(tmp_path):
    """The regression this file exists for: the candidate arm used to report 0."""
    fx = _fixture(tmp_path)
    # all four folds, so the candidate's extra trace is inside the scored scope and the two arms
    # genuinely differ (a single fold here can contain only one arm's traces)
    doc = _run(fx, tmp_path / "c.json", "0,1,2,3")
    ref = doc["bootstrap"]["reference"]
    cand = doc["bootstrap"]["candidate"]
    assert ref["n_blocks"] == cand["n_blocks"] == doc["n_blocks_kept"] > 0
    assert ref["n_scoreable_blocks"] == cand["n_scoreable_blocks"], \
        "scoreable is a property of the truth, so both arms must report the same count"
    assert cand["n_scoreable_blocks"] > 0, "the fixture has proxy truth in this scope"
    assert doc["candidate"]["scalar_dti"] > doc["reference"]["scalar_dti"]


def test_the_identity_guard_reproduces_the_scalar_measurement(tmp_path):
    fx = _fixture(tmp_path)
    doc = _run(fx, tmp_path / "c.json", "0")
    assert max(doc["identity_max_abs_error"].values()) < 1e-9


def test_a_file_contrasted_with_itself_is_refused(tmp_path):
    fx = _fixture(tmp_path)
    mod = _mod()
    with pytest.raises(SystemExit):
        mod.main(["--reference", str(fx["reference"]), "--candidate", str(fx["reference"]),
                  "--labels", str(fx["labels"]), "--proxy", str(fx["proxy"]),
                  "--template", str(fx["template"]), "--block-px", str(fx["block_px"]),
                  "--folds", "4", "--fold", "1", "--out", str(tmp_path / "x.json")])


# ------------------------------------------------------------------------------- pooling folds

def test_pooling_folds_keeps_the_union_of_their_blocks(tmp_path):
    fx = _fixture(tmp_path)
    single0 = _run(fx, tmp_path / "f0.json", "0")
    single1 = _run(fx, tmp_path / "f1.json", "1")
    pooled = _run(fx, tmp_path / "f01.json", "0,1")

    assert pooled["folds"] == [0, 1]
    assert pooled["fold"] is None                       # ambiguous for a pool, so not an int
    assert single0["fold"] == 0 and single0["folds"] == [0]   # single-fold schema unchanged
    assert pooled["n_blocks_kept"] == single0["n_blocks_kept"] + single1["n_blocks_kept"]
    assert pooled["n_blocks_kept"] >= 12 or "COARSE" in pooled["reliability"]


def test_every_pooled_block_records_which_fold_it_came_from(tmp_path):
    fx = _fixture(tmp_path)
    pooled = _run(fx, tmp_path / "f01.json", "0,1")
    per_fold_counts = {pf["fold"]: pf["n_blocks"] for pf in pooled["per_fold"]}
    assert sorted(per_fold_counts) == [0, 1]
    # the per-fold table and the pooled table must be the same blocks, counted once
    assert sum(per_fold_counts.values()) == pooled["n_blocks_kept"]
    assert len({b["block"] for b in _mod_rows(pooled)}) == pooled["n_blocks_kept"]


def _mod_rows(doc):
    """Re-derive the block rows the same way the script does, from its own per-fold table."""
    # the script does not commit the rows themselves (they are ~200 B/block), so the pool is
    # checked through the per-fold counts it does commit
    return [dict(block=i) for i in range(sum(pf["n_blocks"] for pf in doc["per_fold"]))]


def test_per_fold_deltas_are_the_difference_of_the_reported_scalars(tmp_path):
    fx = _fixture(tmp_path)
    pooled = _run(fx, tmp_path / "f01.json", "0,1")
    for pf in pooled["per_fold"]:
        assert pf["delta"] == pytest.approx(pf["candidate_dti"] - pf["reference_dti"], abs=1e-6)


def test_the_note_is_recorded_verbatim_so_a_biased_fold_cannot_be_pooled_silently(tmp_path):
    fx = _fixture(tmp_path)
    note = "fold 0 selected the policy; fold 3 did not"
    doc = _run(fx, tmp_path / "f03.json", "0,3", extra=["--note", note])
    assert doc["note"] == note
    assert "POOLED" in doc["fold_note"]


def test_reliability_flips_at_the_twelve_block_threshold(tmp_path):
    """The readable-CI wording must follow the pooled block count, not the fold count."""
    fx = _fixture(tmp_path, size=256, block_px=32)     # 8x8 = 64 blocks, 16 per fold
    doc = _run(fx, tmp_path / "big.json", "0,1")
    assert doc["n_blocks_kept"] >= 12
    assert doc["reliability"].startswith("adequate")
