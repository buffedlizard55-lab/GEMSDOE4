#!/usr/bin/env python3
"""The metric's own emission budget: how much coverage, at what false-positive price.

WHY THIS EXISTS
---------------
Every policy question in this repository - floor, band width, union vote, how many pixels to emit -
is a question about ONE trade-off, and the official metric defines it exactly.  Write
``TP = TP_w``, ``FP = FP_w``, ``G = |G|`` (the number of ground-truth fault pixels) and
``alpha = 0.2``, ``beta = 0.8``.  Because ``FN_w = sum_g (1 - credit_g) = G - TP`` holds for ANY
prediction whose values lie in [0, 1] (each ground-truth pixel contributes at most 1 through the
max, which is exactly ``k(0) = 1``), the definition

    DTI = TP / (TP + alpha * FP + beta * FN)

(plus the scorer's 1e-7 regulariser, kept explicit because it is part of the number the repository
quotes; it is negligible at grid scale and NOT negligible in a 64 x 64 unit test)

collapses to the two-term identity

    DTI = TP / (alpha * (TP + FP) + beta * G + EPS)                            (IDENTITY)

The prediction enters through TWO numbers only: the weighted coverage ``TP`` and the weighted
false-positive mass ``FP``.  Nothing else about the raster matters.  So "which pixels should we
emit" has a single answer-shaped target: raise ``TP`` without raising ``FP`` more.

This script

  1. VERIFIES the identity numerically against `src.metrics.GtContext` on the raster it is given
     (it is an algebraic consequence of the official formulas, so a disagreement means one of the
     two implementations is wrong - this is a check, not a claim);
  2. reports where the shipped artifact sits in (coverage, FP price) space, on the honest pool
     (folds 0+1: the only geography the NFF members never trained on) and on the whole grid;
  3. inverts the identity: for a target DTI, how much coverage is needed at the current FP price,
     and how much the FP price must fall at the current coverage.

WHAT IT IS NOT.  This is arithmetic on the surrogate truth (the SGMC proxy compilation).  A target
computed here is a target on the SURROGATE, not a leaderboard prediction; ``beta/alpha = 4`` is the
reason coverage is the expensive axis, and that ratio is quoted from the official rules, not chosen
here.

Usage:
    python scripts/emission_budget.py --pred data/evidence/combined/submission.tif
    python scripts/emission_budget.py --pred <raster> --targets 0.25,0.3049,0.35
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import assign_folds, block_table, scored_mask  # noqa: E402
from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS, EPS, GtContext,  # noqa: E402
                         score_within_mask)
from src.submission_io import sha256_file  # noqa: E402

IDENTITY_TOL = 1e-9


def required_coverage(target: float, r: float, alpha: float = DEFAULT_ALPHA,
                      beta: float = DEFAULT_BETA) -> float | None:
    """Coverage TP/G needed for `target` DTI at a fixed FP price r = FP/G.

    From IDENTITY: target = c / (alpha*(c + r) + beta)  =>  c = target*(alpha*r + beta)/(1 - alpha*target)
    Returns None when the target is unreachable at ANY coverage, which happens only for
    target >= 1/alpha = 5 (the metric's own ceiling is 1/alpha, approached as FP, G -> 0).
    """
    denom = 1.0 - alpha * target
    if denom <= 0:
        return None
    return target * (alpha * r + beta) / denom


def required_fp_price(target: float, c: float, alpha: float = DEFAULT_ALPHA,
                      beta: float = DEFAULT_BETA) -> float | None:
    """FP price r = FP/G allowed for `target` DTI at a fixed coverage c.

    From IDENTITY: r = (c*(1 - alpha*target) - beta*target) / (alpha*target).
    Returns None when that is NEGATIVE, which is the informative case: the target is out of reach
    at this coverage even with ZERO false positives, because the best possible DTI at coverage c
    and FP = 0 is c/(alpha*c + beta).  A 0.0 here would read as "met exactly" instead of
    "unreachable", which is the opposite conclusion.
    """
    if target <= 0:
        return None
    r = (c * (1.0 - alpha * target) - beta * target) / (alpha * target)
    return None if r < 0 else r


def max_dti_at_full_coverage(r: float, alpha: float = DEFAULT_ALPHA,
                             beta: float = DEFAULT_BETA) -> float:
    """The ceiling at this FP price: a prediction that covered EVERY truth pixel (c = 1)."""
    return 1.0 / (alpha * (1.0 + r) + beta)


def budget_table(components: dict, targets, alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA) -> list[dict]:
    """Where the prediction is now, and what each target would cost on each axis."""
    TP, FP, G = components["TP_w"], components["FP_w"], components["n_gt"]
    if not G:
        return []
    c, r = TP / G, FP / G
    dti = components["dti"]
    rows = []
    for t in targets:
        need_c = required_coverage(t, r, alpha, beta)
        need_r = required_fp_price(t, c, alpha, beta)
        rows.append(dict(
            target=float(t),
            coverage_axis_reachable=(need_c is not None and need_c <= 1.0),
            coverage_needed=None if need_c is None else float(need_c),
            coverage_gap=None if need_c is None else float(need_c - c),
            fp_axis_reachable=(need_r is not None),
            fp_price_allowed=None if need_r is None else float(need_r),
            fp_price_reduction=None if need_r is None else float(r - need_r),
            fp_price_reduction_frac=None if need_r is None else (
                float((r - need_r) / r) if r > 0 else 0.0),
            already_met=bool(dti >= t),
        ))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="data/evidence/combined/submission.tif")
    ap.add_argument("--truth", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--fold", type=int, default=0, help="selection fold (must match the artifact)")
    ap.add_argument("--eval-fold", type=int, default=1, help="held-out fold")
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", default="0.25,0.3049,0.35,0.40")
    ap.add_argument("--out", default="data/evidence/emission_budget.json")
    args = ap.parse_args(argv)

    targets = [float(x) for x in str(args.targets).split(",") if x.strip()]
    with rasterio.open(args.pred) as s:
        pred = s.read(1)
        profile = s.profile
    with rasterio.open(args.template) as s:
        template = s.read(1)
    with rasterio.open(args.truth) as s:
        truth_codes = s.read(1)
    with rasterio.open(args.labels) as s:
        lab = s.read(1)
        nodata = s.nodata

    footprint = np.isfinite(template)
    proxy = (truth_codes == 2) & footprint
    valid = np.ones(lab.shape, bool) if nodata is None else (lab != nodata)
    fault = (lab == 1) & valid
    ctx = GtContext(proxy, DEFAULT_R_PIXELS)

    table = block_table(fault.shape, args.block_px, valid=footprint, labels=fault)
    fold_of = assign_folds(table, args.folds, args.seed, "balanced")
    sel = scored_mask(fault.shape, args.block_px, fold_of, args.fold)
    mea = scored_mask(fault.shape, args.block_px, fold_of, args.eval_fold)
    scopes = {"selection_fold": sel, "measurement_fold": mea,
              "pooled_folds_0_1": sel | mea, "whole_grid": np.ones(fault.shape, bool)}

    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script": "scripts/emission_budget.py",
           "purpose": ("derive, from the official metric, the coverage/false-positive price of a "
                       "target DTI, and locate the shipped artifact in that space"),
           "identity": {
               "formula": "DTI = TP_w / (alpha*(TP_w + FP_w) + beta*|G| + eps)",
               "eps": EPS,
               "derivation": ("FN_w = sum_g (1 - credit_g) = |G| - TP_w for any prediction in "
                              "[0,1] because credit_g <= k(0) = 1; substituting into "
                              "DTI = TP_w/(TP_w + alpha*FP_w + beta*FN_w) gives the two-term form"),
               "alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA,
               "beta_over_alpha": DEFAULT_BETA / DEFAULT_ALPHA,
               "consequence": ("the prediction enters only through the weighted coverage TP_w and "
                               "the weighted false-positive mass FP_w; one unit of coverage is "
                               "worth beta/alpha = 4 units of false-positive mass")},
           "targets": targets,
           "pred": {"path": args.pred, "sha256": sha256_file(Path(args.pred)),
                    "bytes": Path(args.pred).stat().st_size,
                    "finite_px": int(np.isfinite(pred).sum()),
                    "nonzero_px": int((np.nan_to_num(pred, nan=0.0) > 0).sum())},
           "truth": {"path": args.truth, "pixels": int(proxy.sum()),
                     "population": "SGMC proxy compilation (code 2) - a stand-in, not the scored set"},
           "scopes": {}}

    print(f"[budget] {args.pred}")
    print(f"[budget] identity  DTI = TP_w / (alpha*(TP_w + FP_w) + beta*|G| + eps)   "
          f"alpha={DEFAULT_ALPHA} beta={DEFAULT_BETA}")
    for name, mask in scopes.items():
        s = score_within_mask(pred, ctx, mask)
        if not s["n_gt"]:
            out["scopes"][name] = dict(scoreable=False)
            print(f"[budget]   {name:<18} no truth pixels in this scope - not scoreable")
            continue
        G = s["n_gt"]
        # the identity, checked rather than asserted
        lhs = s["dti"]
        rhs = s["TP_w"] / (DEFAULT_ALPHA * (s["TP_w"] + s["FP_w"]) + DEFAULT_BETA * G + EPS)
        rel = abs(lhs - rhs) / max(abs(lhs), 1e-12)
        fp_price = s["FP_w"] / G
        comp = dict(dti=s["dti"], TP_w=s["TP_w"], FP_w=s["FP_w"], FN_w=s["FN_w"], n_gt=G,
                    pred_px=s["pred_px"], coverage=s["TP_w"] / G, fp_price=fp_price,
                    identity_rhs=rhs, identity_rel_err=rel,
                    identity_holds=bool(rel <= IDENTITY_TOL),
                    ceiling_at_this_fp_price=max_dti_at_full_coverage(fp_price))
        comp["budget"] = budget_table(comp, targets)
        out["scopes"][name] = comp
        flag = "OK " if comp["identity_holds"] else "FAIL"
        print(f"[budget]   {name:<18} DTI {s['dti']:.4f}  coverage {comp['coverage']:.1%}  "
              f"FP/G {comp['fp_price']:.2f}  ceiling-at-this-price {comp['ceiling_at_this_fp_price']:.3f}"
              f"  identity {flag} (rel err {rel:.1e})")
        for b in comp["budget"]:
            if b["already_met"]:
                print(f"[budget]      target {b['target']:.4f}: already met here")
                continue
            nc, nr = b["coverage_needed"], b["fp_price_allowed"]
            cov_txt = ("needs coverage > 100%" if nc is None or not b["coverage_axis_reachable"]
                       else f"{nc:.1%} ({b['coverage_gap']:+.1%})")
            fp_txt = ("unreachable at this coverage" if nr is None
                      else f"{nr:.2f} ({b['fp_price_reduction_frac']:.0%} less)")
            print(f"[budget]      target {b['target']:.4f}: coverage {cov_txt}"
                  f"  |  FP/G allowed {fp_txt}")

    if not all(v.get("identity_holds", True) for v in out["scopes"].values()):
        sys.exit("[budget] FAIL: the identity does not reproduce the scorer - do not quote this file")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"[budget] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
