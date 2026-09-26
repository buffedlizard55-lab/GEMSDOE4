"""Tests for scripts/emission_budget.py — the metric's own coverage / false-positive budget.

WHY THIS FILE EXISTS.  `emission_budget.py` turns the competition metric from "a number a
submission receives" into "a constraint the next submission has to satisfy": because
``TP_w + FN_w = |G|`` for ANY prediction in [0, 1], the distance-weighted Tversky index collapses to

    DTI = TP_w / (alpha * (TP_w + FP_w) + beta * |G| + EPS)

so a prediction enters through exactly two numbers.  Everything the repository does about emission
width, floors and union votes is a question about those two numbers, and a mistake in this arithmetic
would silently mislead every decision built on it.  The tests below therefore pin, in order:

1. the identity itself, against ``src.metrics.GtContext`` (the implementation that scores the
   leaderboard-facing artifacts), at the SAME ``EPS = 1e-7`` the scorer uses - a regulariser that is
   negligible on a 12-million-pixel raster and NOT negligible on a 64-pixel fixture;
2. the two inversions (coverage needed at a given FP price; FP price allowed at a given coverage),
   as round-trips rather than as remembered constants, including the case where the coverage axis
   cannot reach the target at all;
3. the reading of the table: alpha = 0.2 against beta = 0.8 means one more unit of coverage is worth
   four units of false-positive mass - the asymmetry the whole emission policy is built on;
4. the end-to-end script on a synthetic raster, including that it REFUSES to invent a number for a
   target whose FP axis is unreachable (``None``, not 0.0 - printing 0.0 would read as "allowed:
   zero false positives", which is a different and much weaker statement than "not attainable").
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

from src.metrics import DEFAULT_ALPHA, DEFAULT_BETA, EPS, GtContext  # noqa: E402

SCRIPT = ROOT / "scripts" / "emission_budget.py"


def _mod():
    spec = importlib.util.spec_from_file_location("emission_budget", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------------- the identity
def test_the_identity_holds_against_the_scorer_with_its_own_regulariser():
    """DTI == TP / (alpha*(TP + FP) + beta*|G| + EPS) on the implementation that scores artifacts.

    The ``+ EPS`` is not decoration: the scorer's DTI denominator ends in ``+ EPS`` (src/metrics.py),
    so a two-term version of this identity disagrees with it by ~1e-10 at fixture scale.  Asserting
    at 1e-12 relative error is exactly the assertion that would have caught that.
    """
    m = _mod()
    rng = np.random.default_rng(7)
    truth = np.zeros((48, 48), bool)
    truth[10, 5:40] = True
    truth[30:34, 20] = True
    ctx = GtContext(truth, 3)
    for scale in (1.0, 0.5, 0.05):
        pred = (rng.random((48, 48)) * scale).astype(np.float32)
        pred[10, 5:40] = 1.0
        dti, (tp, fp, fn) = ctx.score(pred, return_components=True)
        # |G| IS TP_w + FN_w, and that is what makes the scorer's three-term denominator equal the
        # two-term form: TP + a*FP + b*FN == a*(TP + FP) + b*|G|.
        n_gt = tp + fn
        assert n_gt == pytest.approx(float(truth.sum()), abs=1e-9), \
            "TP_w + FN_w = |G| must hold for any prediction in [0, 1]"
        lhs = tp / (DEFAULT_ALPHA * (tp + fp) + DEFAULT_BETA * n_gt + EPS)
        assert lhs == pytest.approx(dti, rel=1e-12), f"identity broken at scale {scale}"


def test_the_two_term_identity_needs_the_epsilon():
    """Guard against 'simplifying' EPS away: the difference is real at fixture scale."""
    m = _mod()
    rng = np.random.default_rng(3)
    truth = np.zeros((32, 32), bool)
    truth[5, :] = True
    ctx = GtContext(truth, 3)
    pred = (rng.random((32, 32)) * 0.2).astype(np.float32)
    pred[5, :] = 1.0
    dti, (tp, fp, fn) = ctx.score(pred, return_components=True)
    without = tp / (DEFAULT_ALPHA * (tp + fp) + DEFAULT_BETA * (tp + fn))
    assert abs(without - dti) > 1e-11, \
        "this fixture no longer discriminates the EPS term - pick a smaller fixture"


# ------------------------------------------------------------------------------------- inversions
def test_the_inversions_are_round_trips_not_remembered_constants():
    m = _mod()
    for c in (0.05, 0.2, 0.55, 0.9):
        for r in (0.0, 0.3, 2.0, 8.0):
            dti = c / (DEFAULT_ALPHA * (c + r) + DEFAULT_BETA + EPS / 1.0 * 0.0)
            dti = c / (DEFAULT_ALPHA * (c + r) + DEFAULT_BETA)
            assert m.required_coverage(dti, r) == pytest.approx(c, rel=1e-9)
            assert m.required_fp_price(dti, c) == pytest.approx(r, rel=1e-9)


def test_an_unreachable_false_positive_axis_is_none_not_zero():
    """At small coverage even ZERO false positives cannot reach a high target.

    The zero-FP ceiling is c / (alpha*c + beta) (set r = 0 above), which is below 0.25 for any
    coverage under ~15 %.  Reading that as "0.00 FP allowed" would tell a planner that precision
    alone can get there; it cannot.  The function must say so with None.
    """
    m = _mod()
    assert m.max_dti_at_full_coverage(0.0) == pytest.approx(1.0 / (DEFAULT_ALPHA + DEFAULT_BETA))
    for target in (0.25, 0.30, 0.35):
        assert m.required_fp_price(target, 0.10) is None, \
            f"coverage 10% cannot reach {target} even at zero false positives"
    # and the coverage needed for a target at a fixed 5% FP price is a real, reachable fraction
    need = m.required_coverage(0.30, 0.05)
    assert 0.0 < need < 1.0
    # 1.5 DTI at zero FP price would need 1.5*0.8/(1 - 0.2*1.5) = 1.71 x the truth covered
    assert m.required_coverage(1.5, 0.0) > 1.0, \
        "an impossible target must come back as >100% coverage, not as a clipped number"
    # the reachable-but-out-of-reach-at-this-coverage case: 10% coverage, target 0.80
    assert m.required_fp_price(0.80, 0.10) is None


def test_the_axes_read_the_way_a_planner_needs():
    """Hand arithmetic on one components dict, so the printed table cannot drift from the formulas.

    alpha = 0.2, beta = 0.8.  A missed fault pixel therefore costs FOUR times an emitted
    false-positive pixel, which is why every policy in this repository trades precision for recall.
    """
    m = _mod()
    #   coverage c = 0.4 (40 of 100 truth px covered), FP price r = 0.2 (20 priced px)
    comp = dict(dti=40 / (0.2 * (40 + 20) + 0.8 * 100), TP_w=40.0, FP_w=20.0, n_gt=100)
    rows = m.budget_table(comp, targets=(0.3049, 0.50, 0.80))
    by_target = {r["target"]: r for r in rows}
    # c needed at the current price r = FP/G: target = c/(alpha*(c+r)+beta)  =>
    #   c = target*(alpha*r + beta) / (1 - alpha*target)
    need = by_target[0.3049]["coverage_needed"]
    assert need == pytest.approx(0.3049 * (0.2 * 0.2 + 0.8) / (1 - 0.2 * 0.3049), rel=1e-9)
    # ... and the price allowed at the current coverage c: r = (c(1-alpha*t) - beta*t)/(alpha*t)
    assert by_target[0.3049]["fp_price_allowed"] == pytest.approx(
        (0.4 * (1 - 0.2 * 0.3049) - 0.8 * 0.3049) / (0.2 * 0.3049), rel=1e-9)
    assert by_target[0.3049]["coverage_axis_reachable"] is True
    assert by_target[0.3049]["fp_price_allowed"] is not None
    # a target above the zero-FP ceiling at this coverage has NO FP price that reaches it
    assert by_target[0.80]["coverage_axis_reachable"] is True
    assert by_target[0.80]["coverage_needed"] == pytest.approx(
        0.80 * (0.2 * 0.2 + 0.8) / (1 - 0.2 * 0.80), rel=1e-9)
    assert by_target[0.80]["fp_price_allowed"] is None, \
        "0.80 at 40 % coverage allows no false positives at all - the FP axis is the one that dies"

    # an expensive regime: 10 covered px, 200 emitted px, 100 truth px
    # an expensive regime: 10 of 100 truth px covered and 200 emitted px (price 2.0)
    hard = m.budget_table(dict(dti=10 / (0.2 * (10 + 200) + 0.8 * 100), TP_w=10.0, FP_w=200.0,
                               n_gt=100), targets=(0.30, 0.80))
    assert hard[0]["coverage_axis_reachable"] is True and hard[0]["coverage_needed"] < 1.0
    assert hard[1]["coverage_axis_reachable"] is False, \
        "at this price 0.80 needs more than all the truth covered"
    assert hard[1]["coverage_needed"] > 1.0


# ------------------------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    d = tmp_path_factory.mktemp("budget")
    rng = np.random.default_rng(11)
    H = W = 128
    inside = np.zeros((H, W), bool)
    inside[8:H - 8, 8:W - 8] = True
    labels = np.full((H, W), -1, np.int8)
    labels[inside] = 0
    labels[20, 10:110] = 1
    labels[60:62, 30:100] = 1
    template = np.where(inside, 0.0, np.nan).astype(np.float32)
    proxy = np.zeros((H, W), np.uint8)
    proxy[90, 10:110] = 2
    proxy[40:42, 40:90] = 2
    pred = np.zeros((H, W), np.float32)
    pred[20, 10:110] = 1.0
    pred[(rng.random((H, W)) < 0.02) & inside] = 1.0
    pred[~inside] = np.nan

    def _write(name, arr, nodata=None):
        p = d / name
        with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1,
                           dtype=arr.dtype, crs="EPSG:32611",
                           transform=from_origin(0, H * 100, 100, 100), nodata=nodata) as dst:
            dst.write(arr, 1)
        return p

    return dict(labels=_write("labels.tif", labels, -1), template=_write("sample.tif", template,
                                                                        float("nan")),
                proxy=_write("proxy.tif", proxy), pred=_write("pred.tif", pred, float("nan")),
                out=d / "budget.json")


def test_the_script_writes_a_budget_it_verified_first(fixture):
    m = _mod()
    rc = m.main(["--pred", str(fixture["pred"]), "--truth", str(fixture["proxy"]),
                 "--labels", str(fixture["labels"]), "--template", str(fixture["template"]),
                 "--folds", "4", "--block-px", "32", "--targets", "0.30,0.45",
                 "--out", str(fixture["out"])])
    assert rc == 0
    doc = json.loads(fixture["out"].read_text())
    assert doc["identity"]["eps"] == EPS
    assert doc["identity"]["beta_over_alpha"] == 4.0
    for scope, d in doc["scopes"].items():
        assert d["identity_holds"] is True, f"{scope}: the identity must reproduce the scorer"
        assert d["identity_rel_err"] < 1e-9
    assert set(doc["scopes"]) >= {"selection_fold", "measurement_fold", "pooled_folds_0_1",
                                  "whole_grid"}
    for scope, d in doc["scopes"].items():
        assert d["FP_w"] >= 0.0 and d["TP_w"] >= 0.0
        assert 0.0 <= d["coverage"] <= 1.0 and d["fp_price"] >= 0.0
        assert d["n_gt"] > 0
        for row in d["budget"]:
            assert row["target"] in (0.30, 0.45)
            assert row["fp_price_allowed"] is None or row["fp_price_allowed"] >= 0.0
            if row["fp_price_allowed"] is not None:
                # an allowed price must actually reach the target at this coverage, and a reachable
                # COVERAGE axis must be a fraction
                reached = d["coverage"] / (DEFAULT_ALPHA * (d["coverage"]
                                                            + row["fp_price_allowed"])
                                           + DEFAULT_BETA)
                assert reached == pytest.approx(row["target"], rel=1e-6)
                assert row["coverage_axis_reachable"] is True
            assert row["fp_axis_reachable"] is (row["fp_price_allowed"] is not None)
