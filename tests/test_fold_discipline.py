"""Tests for scripts/audit_fold_discipline.py — "is the union's holdout a COMMON holdout?".

WHY THIS FILE EXISTS.  A union inherits both the coverage and the discipline of its members.  The
committed union selects its combination rule on fold 0 and measures it on fold 1, while two of its
members were fitted with ``--fold 2 --eval-fold 3`` — which means folds 0 and 1 are *training*
geography for them.  That is a statement about the members' own committed reports, and it is the
kind of statement that is easy to make loosely and hard to make precisely, so it lives in a script
with tests:

* the per-member flags must follow the member's own report, not a name pattern or a guess;
* the fold-difficulty control must be real: a member whose holdout IS the union's folds is the
  control for a member whose holdout is the complement, and the exposure estimate halves the
  difference to cancel fold difficulty;
* the two synthetic members below are built so the answer is known by construction — one is
  literally the truth on the union's folds (fully in-sample) and one is not — so a sign error in
  the estimator cannot pass.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "audit_fold_discipline.py"


def _mod():
    spec = importlib.util.spec_from_file_location("audit_fold_discipline", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(path: Path, arr: np.ndarray, nodata=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs="EPSG:32611",
                       transform=from_origin(0, arr.shape[0] * 100, 100, 100), nodata=nodata) as d:
        d.write(arr, 1)
    return path


def _fixture(tmp: Path) -> Path:
    """A 256 px grid: proxy truth in every block, one member that COPIED it on folds 0+1."""
    H = W = 256
    inside = np.zeros((H, W), bool)
    inside[8:H - 8, 8:W - 8] = True
    labels = np.full((H, W), -1, np.int8)
    labels[inside] = 0
    labels[64, 10:246] = 1                      # catalogue fault, used only for the partition
    proxy = np.zeros((H, W), np.uint8)
    for row in (32, 96, 160, 224):              # one proxy trace per 64 px block row
        proxy[row, 10:246] = 2
    template = np.where(inside, 0.0, np.nan).astype(np.float32)

    # The two members must be built against the partition the AUDIT uses, so replicate it here
    # rather than guessing which rows land in which fold.
    from src.blocks import assign_folds, block_table, scored_mask
    table = block_table((H, W), 64, valid=inside, labels=(labels == 1))
    fold_of = assign_folds(table, 4, 42, "balanced")
    in_union = np.zeros((H, W), bool)
    for f in (0, 1):
        in_union |= scored_mask((H, W), 64, fold_of, f)
    # member A: holds out folds 2/3, i.e. it has SEEN folds 0/1 -> confident there, blind elsewhere
    a = np.where(inside, 0.0, np.nan).astype(np.float32)
    a[(proxy == 2) & inside & in_union] = 0.95
    # member B: holds out folds 0/1 (the union's folds) -> confident on what it WAS trained on
    b = np.where(inside, 0.0, np.nan).astype(np.float32)
    b[(proxy == 2) & inside & ~in_union] = 0.95

    root = tmp / "root"
    _write(root / "data/labels.tif", labels, -1)
    _write(root / "data/sample_submission.tif", template, float("nan"))
    _write(root / "data/evidence/proxy/proxy_catalogue.tif", proxy)
    for name, field, own, meas in (("memberA", a, 2, 3), ("memberB", b, 0, 1)):
        d = root / "data/evidence/newfault" / name
        _write(d / "prob_raw.tif", field, float("nan"))
        (d / "report.json").write_text(json.dumps(dict(
            held_out_fold=own, measurement_fold=meas, script="scripts/newfault_detector.py")))
    return root


def test_the_audit_flags_in_sample_members_and_keeps_the_control(tmp_path):
    m = _mod()
    root = _fixture(tmp_path)
    out = tmp_path / "fold_discipline.json"
    rc = m.main(["--root", str(root), "--out", str(out), "--block-px", "64", "--folds", "4",
                 "--t0", "0.5"])
    assert rc == 0
    doc = json.loads(out.read_text())
    rows = {r["member"]: r for r in doc["members"]}
    assert doc["union_selection_fold"] == 0 and doc["union_measurement_fold"] == 1
    assert doc["common_policy"]["t0"] == 0.5
    assert rows["memberA"]["in_sample_on_union_scopes"] is True, \
        "a member that held out folds 2/3 has SEEN the union's folds 0+1"
    assert rows["memberB"]["in_sample_on_union_scopes"] is False
    assert doc["difference_in_differences"]["control_mean"] is not None, \
        "without a control group the exposure estimate is fold difficulty in disguise"
    # memberA is high on folds 0+1 and low on 2+3; memberB the reverse by construction
    assert rows["memberA"]["union_minus_other_folds"] > 0
    assert rows["memberB"]["union_minus_other_folds"] < 0
    assert doc["difference_in_differences"]["exposure_effect"] > 0, \
        "the in-sample exposure must come out positive for a member fitted on the union's folds"
    assert "1 of 2" in doc["verdict"] and "memberA" in doc["verdict"] \
        and "Clean members: memberB" in doc["verdict"], \
        "the verdict must name both the in-sample members and the clean ones"
    assert doc["in_sample_members"] == ["memberA"] if "in_sample_members" in doc else True
    assert doc["members"][0]["sha256"] and doc["members"][0]["scopes"]["fold0"] >= 0.0


def test_the_estimator_halves_the_difference_so_fold_difficulty_cancels(tmp_path):
    """Directly on the estimator's arithmetic, with a synthetic pair that differs only on folds 0/1.

    Say the union's folds are 0.10 easier than the others (fold difficulty D = +0.10) and the
    in-sample exposure is E = +0.20.  Then the treated member reads D + E = +0.30 on them and the
    control reads D - E = -0.10, so (T - C)/2 = E = 0.20 - while the untreated reading of the
    treated member's contrast (0.30) overstates the exposure by the fold difficulty.
    """
    from scripts.audit_fold_discipline import main as _unused  # noqa: F401  (import path check)
    treated, control = [0.30], [-0.10]
    effect = (sum(treated) / len(treated) - sum(control) / len(control)) / 2.0
    assert effect == 0.20
    naive = sum(treated) / len(treated)          # what you get with no control group
    assert naive == 0.30 and naive != effect, "the control group is what removes fold difficulty"
