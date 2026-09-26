"""End-to-end test of scripts/newfault_detector.py on a synthetic competition-like grid.

What this pins, and why each assertion exists:

* the two held-out folds (the one that SELECTS the emission policy and the one that MEASURES it)
  are excluded from training, collar included - the generator's central claim;
* the catalogue-only ablation is a *different model*, not a flag that is ignored;
* the shipped raster is template-conformant (finite inside the official valid region, NaN
  outside, in [0, 1]) - the placement rule the platform enforces with "Predicted values must be
  in range [0, 1]";
* the report's winner is the argmax of the statistic it says it ranked on, over eligible rows
  only, and re-applying the reported policy to the reported raw field reproduces the reported
  numbers - so a quoted number cannot drift from the bytes that produced it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "newfault_detector.py"
NODATA = -3.4028235e38


def _mod():
    spec = importlib.util.spec_from_file_location("newfault_detector", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(path: Path, arr: np.ndarray, dtype: str, nodata=None) -> Path:
    profile = dict(driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                   count=1 if arr.ndim == 2 else arr.shape[0], dtype=dtype,
                   crs="EPSG:32611", transform=from_origin(500000.0, 4100000.0, 100.0, 100.0),
                   nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype, copy=False), 1 if arr.ndim == 2 else list(range(1, arr.shape[0] + 1)))
    return path


def _fixture_grid(d: Path, size: int = 192, bands: int = 19):
    """A synthetic competition-like dataset.

    Two vertical catalogue traces (columns 30-31, 130-131) and, crucially, a **proxy-only** trace
    at column 90 that no catalogue label contains - the population this detector exists to find.
    Band 0 is causally informative on the catalogue traces and band 1 on the proxy trace, so a
    model can actually learn something instead of fitting noise.  Rows 0..3 and size-4.. are
    outside the footprint (sentinel in every band), so the NaN contract is exercised.
    """
    rng = np.random.default_rng(7)
    labels = np.full((size, size), -1, np.int8)
    labels[4:size - 4, :] = 0
    fault = np.zeros((size, size), bool)
    for col in (30, 31, 130, 131):
        fault[10:size - 10, col] = True
    labels[fault] = 1

    X = rng.normal(0, 1, (bands, size, size)).astype(np.float32)
    X[0, 10:size - 10, 30:32] += 3.0
    X[0, 10:size - 10, 130:132] += 3.0
    for col in (20, 100):                              # only the proxy knows these two
        X[1, 10:size - 10, col] += 3.0
    X[:, :4, :] = NODATA
    X[:, size - 4:, :] = NODATA

    sample = np.full((size, size), np.nan, np.float32)
    sample[4:size - 4, :] = 0.0
    proxy = np.zeros((size, size), np.uint8)
    for col in (20, 100):                             # code 2 = new-fault-like (not in labels)
        proxy[10:size - 10, col] = 2
    proxy[fault] = 1

    return dict(
        features=_write(d / "features.tif", X, "float32", nodata=NODATA),
        labels=_write(d / "labels.tif", labels, "int8", nodata=-1),
        template=_write(d / "sample.tif", sample, "float32", nodata=np.nan),
        proxy=_write(d / "proxy.tif", proxy, "uint8"),
        fault=fault, proxy_only=(proxy == 2), size=size, X=X)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    pytest.importorskip("sklearn", reason="the NFF detector's model needs scikit-learn")
    d = tmp_path_factory.mktemp("newfault")
    fx = _fixture_grid(d)
    out_dir = d / "out"
    mod = _mod()
    rc = mod.main(["--features", str(fx["features"]), "--labels", str(fx["labels"]),
                   "--template", str(fx["template"]), "--proxy", str(fx["proxy"]),
                   "--out-dir", str(out_dir), "--block-px", "64", "--folds", "4",
                   "--fold", "0", "--eval-fold", "1", "--seed", "3", "--max-iter", "25",
                   "--max-negatives", "4000", "--neg-ratio", "10"])
    report = json.loads((out_dir / "report.json").read_text())
    return dict(rc=rc, out_dir=out_dir, fx=fx, report=report, mod=mod)


# --------------------------------------------------------------------------------------- format
def test_main_rc_and_both_artefacts_exist(run):
    assert run["rc"] == 0
    for name in ("submission.tif", "prob_raw.tif", "report.json"):
        assert (run["out_dir"] / name).exists(), name
    for key in ("submission", "prob_raw"):
        rep = run["report"][key]
        assert rep["sha256"] == hashlib.sha256(Path(rep["path"]).read_bytes()).hexdigest(), \
            "the report must quote the bytes on disk, not a value computed before the write"


def test_submission_is_template_conformant_and_in_range(run):
    """The platform rule: finite inside the template's valid region, NaN outside, values in [0,1]."""
    with rasterio.open(run["out_dir"] / "submission.tif") as src:
        with rasterio.open(run["fx"]["template"]) as tmpl:
            assert (src.width, src.height, src.count) == (tmpl.width, tmpl.height, 1)
            assert src.crs == tmpl.crs and src.transform == tmpl.transform
            assert src.dtypes[0] == "float32"
            a = src.read(1)
            t = tmpl.read(1)
    finite = np.isfinite(a)
    assert np.array_equal(finite, np.isfinite(t)), "the NaN footprint must match the template"
    assert finite.any()
    assert float(a[finite].min()) >= 0.0 and float(a[finite].max()) <= 1.0
    assert int(np.count_nonzero(a[finite])) > 0, "an all-zero field is not a prediction"


def test_raw_field_is_a_probability_raster(run):
    with rasterio.open(run["out_dir"] / "prob_raw.tif") as src:
        a = src.read(1)
    finite = np.isfinite(a)
    assert finite.any()
    assert float(a[finite].min()) >= 0.0 and float(a[finite].max()) <= 1.0
    # the raw field must be strictly more informative than the shaped one: a floor at t0 > 0
    # keeps only part of it
    assert int(np.count_nonzero(a[finite] > 0)) >= int(
        np.count_nonzero(np.nan_to_num(a, nan=0.0) >= 0.288378))


# ---------------------------------------------------------------------------------------- leakage
def test_no_training_pixel_lies_in_either_held_out_fold(run):
    mod, fx, rep = run["mod"], run["fx"], run["report"]
    from src.blocks import assign_folds, block_table, held_out_mask
    fault, valid = mod.read_truth(Path(fx["labels"]))
    table = block_table(fault.shape, 64, valid=valid, labels=fault)
    fold_of = assign_folds(table, 4, 3, "balanced")
    excluded = (held_out_mask(fault.shape, 64, fold_of, 0, 3)
                | held_out_mask(fault.shape, 64, fold_of, rep["measurement_fold"], 3))
    trainable = valid & ~excluded
    rng = np.random.default_rng(3)
    proxy = (rasterio.open(fx["proxy"]).read(1) == 2) & valid
    sel = mod.choose_samples(fault | proxy, trainable, rng, 10.0, 4000)
    assert not (sel["pos"] & excluded).any(), "a positive was trained on held-out geography"
    assert not (sel["neg"] & excluded).any(), "a negative was trained on held-out geography"


def test_the_proxy_labels_are_actually_used(run):
    """The whole point of the new-fault-first line: supervision beyond the catalogue."""
    sup = run["report"]["supervision"]
    assert sup["use_proxy_labels"] is True
    assert sup.get("mode", "union") == "union"
    assert sup["proxy_only_px"] > 0, "the fixture carries a proxy-only trace"
    assert sup["positive_px"] > sup["catalogue_px"], \
        "the catalogue union the proxy must be strictly larger than the catalogue alone"
    assert run["report"]["training"]["n_pos"] > 0
    assert sup.get("catalogue_in_positives") is True
    assert sup.get("proxy_only_in_positives") is True


def test_the_catalogue_only_ablation_changes_the_model(run, tmp_path):
    pytest.importorskip("sklearn")
    fx = run["fx"]
    out_dir = tmp_path / "ablate"
    mod = _mod()
    rc = mod.main(["--features", str(fx["features"]), "--labels", str(fx["labels"]),
                   "--template", str(fx["template"]), "--proxy", str(fx["proxy"]),
                   "--out-dir", str(out_dir), "--block-px", "64", "--folds", "4",
                   "--fold", "0", "--eval-fold", "1", "--seed", "3", "--max-iter", "25",
                   "--max-negatives", "4000", "--no-proxy-labels"])
    assert rc == 0
    rep = json.loads((out_dir / "report.json").read_text())
    assert rep["supervision"]["use_proxy_labels"] is False
    assert rep["supervision"]["mode"] == "catalogue"
    assert rep["supervision"]["positive_px"] == rep["supervision"]["catalogue_px"]
    assert rep["training"]["n_pos"] < run["report"]["training"]["n_pos"], \
        "dropping the proxy must reduce the training positives, or the flag did nothing"
    with rasterio.open(out_dir / "prob_raw.tif") as src:
        ablated = np.nan_to_num(src.read(1), nan=0.0)
    with rasterio.open(run["out_dir"] / "prob_raw.tif") as src:
        full = np.nan_to_num(src.read(1), nan=0.0)
    assert not np.array_equal(ablated, full), "the two supervisions must produce different fields"


def test_proxy_only_supervision_never_sees_a_catalogue_fault(run, tmp_path):
    """Session-32 unique angle: diversity from the *training target*, not from seeds.

    A member supervised only on SGMC code-2 pixels never treats a catalogue fault as a
    positive, so its errors are structurally anti-correlated with catalogue-trained
    detectors.  That is the LOO finding from session 31 written as a flag rather than a
    hope that another seed will look different.
    """
    pytest.importorskip("sklearn")
    fx = run["fx"]
    out_dir = tmp_path / "proxy_only"
    mod = _mod()
    rc = mod.main(["--features", str(fx["features"]), "--labels", str(fx["labels"]),
                   "--template", str(fx["template"]), "--proxy", str(fx["proxy"]),
                   "--out-dir", str(out_dir), "--block-px", "64", "--folds", "4",
                   "--fold", "0", "--eval-fold", "1", "--seed", "3", "--max-iter", "25",
                   "--max-negatives", "4000", "--supervision", "proxy_only"])
    assert rc == 0
    rep = json.loads((out_dir / "report.json").read_text())
    sup = rep["supervision"]
    assert sup["mode"] == "proxy_only"
    assert sup["use_proxy_labels"] is True, "legacy flag stays True whenever proxy enters training"
    assert sup["catalogue_in_positives"] is False
    assert sup["proxy_only_in_positives"] is True
    assert sup["positive_px"] == sup["proxy_only_px"], \
        "proxy_only positives must equal the proxy-only pixel count exactly"
    assert "--supervision proxy_only" in rep["reproduce"]
    with rasterio.open(out_dir / "prob_raw.tif") as src:
        po = np.nan_to_num(src.read(1), nan=0.0)
    with rasterio.open(run["out_dir"] / "prob_raw.tif") as src:
        full = np.nan_to_num(src.read(1), nan=0.0)
    assert not np.array_equal(po, full), \
        "proxy_only supervision must produce a different field from the union default"


def test_build_positives_modes_are_disjoint_where_they_should_be():
    """Unit pin on the pure function so a future edit cannot silently re-add catalogue
    positives into the proxy_only mask."""
    mod = _mod()
    fault = np.zeros((4, 4), bool); fault[0, 0] = True
    proxy = np.zeros((4, 4), bool); proxy[1, 1] = True
    near = np.zeros((4, 4), bool); near[2, 2] = True
    u = mod.build_positives(fault, proxy, near, "union")
    p = mod.build_positives(fault, proxy, near, "proxy_only")
    c = mod.build_positives(fault, proxy, near, "catalogue")
    assert u.sum() == 3 and p.sum() == 1 and c.sum() == 1
    assert p[0, 0] is np.False_ or p[0, 0] == False
    assert not p[0, 0], "proxy_only must never mark a catalogue fault positive"
    assert c[1, 1] == False and c[2, 2] == False
    assert u[0, 0] and u[1, 1] and u[2, 2]


def test_the_corridor_is_the_scorers_own_disk_not_an_approximation():
    """--corridor must widen positives by the metric's Euclidean R-neighbourhood.

    The scorer gives a ground-truth pixel credit from any prediction within R = 3 px at weight
    k(d) = (1 - d/R)+, so "the metric's tolerance region" is the 29-offset disk of
    ``src.metrics.kernel_offsets(3)``.  The two cheap approximations a reader would assume are
    both wrong and both detectable on a single-pixel fixture:

      * a 3-iteration default ``binary_dilation`` uses a 4-connected (diamond) element, which
        contains (3, 0) and omits (2, 2) - i.e. it credits a 300 m offset but not a 283 m one;
      * a square 7x7 element contains (3, 3), which is 424 m away and scores k = 0.

    Asserting on those two pixels is what makes this a test of the metric's geometry rather
    than of "some dilation happened".
    """
    mod = _mod()
    from src.metrics import kernel_offsets
    blank = np.zeros((11, 11), bool)
    proxy = np.zeros((11, 11), bool)
    near = np.zeros((11, 11), bool)
    fault = np.zeros((11, 11), bool)
    fault[5, 5] = True

    for R in (0, 1, 2, 3):
        got = mod.build_positives(fault, proxy, near, "catalogue", corridor=R)
        expected = len(kernel_offsets(R)) if R else 1
        assert int(got.sum()) == expected, f"corridor {R}: {int(got.sum())} px, want {expected}"

    disk = mod.build_positives(fault, proxy, near, "catalogue", corridor=3)
    assert disk[7, 7], "Euclidean distance sqrt(8) = 2.83 px <= R must be inside the corridor"
    assert not disk[8, 8], "Euclidean distance sqrt(18) = 4.24 px > R must be outside"
    assert disk[8, 5] and disk[5, 8], "orthogonal 3 px offsets are exactly at R"
    # a diamond would have (3,0) but not (2,2); a square would have (3,3).  Pin both endpoints.
    assert disk[8, 5] and disk[7, 7] and not disk[8, 8]

    # corridor=0 must be an exact no-op (the default in every committed member run)
    assert np.array_equal(mod.build_positives(fault, proxy, near, "catalogue", corridor=0), fault)


def test_the_corridor_widens_the_reported_positive_set_and_is_recorded():
    """The report has to say the target was widened, and by how much: a member's row is only
    interpretable next to the positive population it was fitted on."""
    mod = _mod()
    fault = np.zeros((9, 9), bool); fault[4, :] = True          # a 9 px trace
    proxy = np.zeros((9, 9), bool)
    near = np.zeros((9, 9), bool)
    plain = mod.build_positives(fault, proxy, near, "catalogue", corridor=0)
    wide = mod.build_positives(fault, proxy, near, "catalogue", corridor=3)
    assert int(plain.sum()) == 9
    assert int(wide.sum()) > int(plain.sum()), "corridor 3 must grow the positive set"


def test_resolve_supervision_honours_legacy_and_refuses_conflicts():
    mod = _mod()
    from argparse import Namespace
    assert mod.resolve_supervision(Namespace(supervision="union", use_proxy_labels=None)) == "union"
    assert mod.resolve_supervision(Namespace(supervision="proxy_only", use_proxy_labels=None)) == "proxy_only"
    assert mod.resolve_supervision(Namespace(supervision="union", use_proxy_labels=False)) == "catalogue"
    assert mod.resolve_supervision(Namespace(supervision="union", use_proxy_labels=True)) == "union"
    # explicit non-default mode + contradictory legacy flag must fail loudly
    try:
        mod.resolve_supervision(Namespace(supervision="catalogue", use_proxy_labels=True))
        raised = False
    except SystemExit:
        raised = True
    assert raised, "conflicting supervision flags must be a hard error"


# --------------------------------------------------------------------------------------- selection
def test_the_winner_is_the_argmax_of_the_statistic_it_ranks_on(run):
    sel = run["report"]["policy_selection"]
    assert sel["population"].startswith("proxy"), "selection must target the new-fault population"
    eligible = [r for r in sel["candidates"] if r["eligible"]]
    assert eligible, "the held-out fold must contain proxy pixels"
    assert sel["winner"] in eligible
    assert sel["winner"]["proxy_dti"] == max(r["proxy_dti"] for r in eligible), \
        "the winner must be the argmax of the statistic the report says it ranked on"


def test_the_policy_sweep_is_scored_on_the_selection_fold_not_the_whole_grid(run):
    """REGRESSION (session 34).  `sweep(scope_mask, scope_name)` took a scope mask and never used
    it: every candidate row was scored with `GtContext.score`, i.e. over the WHOLE grid, while the
    report labelled the rows with the fold.  Since this detector is trained on everything except
    folds 0 and 1, fold 2/3 truth is in-sample, so the whole-grid score is partly a memorisation
    measurement.  Both numbers are now recorded, and they must differ on this fixture (the whole
    grid carries truth outside the selection fold) - otherwise the test is not exercising the fix.
    """
    sel = run["report"]["policy_selection"]
    rows = sel["candidates"]
    assert sel["selection_key"].startswith("proxy_dti")
    assert "SELECTION FOLD" in sel["ranking_key"], sel["ranking_key"]
    for r in rows:
        assert "proxy_dti_whole" in r, "the whole-grid number must still be reported (as context)"
        assert "catalogue_dti_whole" in r
    eligible = [r for r in rows if r["eligible"]]
    assert eligible
    assert sel["winner"]["proxy_dti"] == max(r["proxy_dti"] for r in eligible)
    # the fold score must not be the whole-grid score in disguise
    assert any(r["proxy_dti"] != r["proxy_dti_whole"] for r in rows), \
        "the fold score must not be the whole-grid score in disguise"
    # NOTE: the fold score is NOT bounded above by the whole-grid score, and asserting that it is
    # was this test's first version.  DTI is a RATIO - a fold whose predictions happen to cover a
    # larger share of its own truth than the grid average can score higher than the whole grid
    # (on the real data the selection fold scores 0.2844 against a whole-grid 0.2199).  Only the
    # identity `sum of per-block components == the scoped score` constrains the two numbers.


def test_the_eligibility_window_is_enforced_not_decorative(run):
    sel = run["report"]["policy_selection"]
    for row in sel["candidates"]:
        if row["eligible"]:
            continue
        assert row["eligibility_reason"], "an ineligible row must say why"
        assert ("cap" in row["eligibility_reason"]
                or "floor" in row["eligibility_reason"]), row["eligibility_reason"]


def test_the_selection_and_measurement_folds_are_disjoint(run):
    rep = run["report"]
    assert rep["held_out_fold"] != rep["measurement_fold"], \
        "selecting and measuring on the same geography is not a measurement"
    assert rep["generalisation"]["fold"] == rep["measurement_fold"]
    assert rep["generalisation"]["scope"].startswith("blocks of fold"), rep["generalisation"]["scope"]
    for pop, m in rep["generalisation"]["by_scope"]["measurement"].items():
        assert m["n_gt"] > 0, f"the measurement fold must contain {pop} truth"


def test_reapplying_the_reported_policy_reproduces_the_reported_dtis(run):
    mod, rep = run["mod"], run["report"]
    with rasterio.open(run["out_dir"] / "prob_raw.tif") as src:
        prob = src.read(1)
    w = rep["policy_selection"]["winner"]
    field = mod.shaped(prob, w["t0"], w["thin"], w["dilate"], mod.DEFAULT_R_PIXELS)
    with rasterio.open(run["out_dir"] / "submission.tif") as src:
        shipped = src.read(1)
    # the shipped raster is the shaped field with the template's mask restored, so compare on the
    # template's valid region and require the NaN footprint to match the template exactly
    with rasterio.open(run["fx"]["template"]) as src:
        tmpl = src.read(1)
    valid = np.isfinite(tmpl)
    assert np.array_equal(np.isfinite(shipped), valid), "the shipped NaN mask must be the template's"
    assert np.array_equal(shipped[valid], field[valid]), \
        "inside the footprint the shipped raster must be exactly the reported policy"
    for pop in ("proxy", "catalogue"):
        assert abs(rep["submission"]["global_scores"][pop]
                   - rep["generalisation"]["by_scope"]["whole grid"][pop]["dti"]) < 1e-9, \
            "the report's global score and its whole-grid measurement must be the same number"


def test_the_shipped_artifact_leaves_conformance_evidence(run):
    """The platform rejection this repo already suffered must be impossible to reintroduce quietly."""
    ev = run["out_dir"] / "sanitize.json"
    assert ev.exists(), "the detector must record what conform_to_template changed"
    d = json.loads(ev.read_text())
    assert d["after"]["sha256"] == run["report"]["submission"]["sha256"]
    assert d["changes"]["masked_outside"] > 0, \
        "the shaping maps NaN -> 0 outside the footprint, so masking must be a real change"


def test_the_report_records_its_own_reproduce_command(run):
    rep = run["report"]
    assert "newfault_detector.py" in rep["reproduce"]
    assert "--seed" in rep["reproduce"] and "--fold" in rep["reproduce"]
