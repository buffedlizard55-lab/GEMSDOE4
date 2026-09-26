#!/usr/bin/env python3
"""Audit whether the union's evaluation folds are a COMMON holdout for its members.

WHY THIS EXISTS
---------------
A union's fold discipline is only as good as its weakest member.  ``scripts/combine_newfault.py``
selects the combination rule on the *selection fold* and measures it on the *measurement fold*, and
the honest reading of both requires that no member trained on that geography.  Two members of the
shipped union do not satisfy that: ``nff43`` and ``nff45`` were run with ``--fold 2 --eval-fold 3``,
i.e. they excluded folds 2 and 3 from training, which means folds 0 and 1 are IN-SAMPLE for them
(``scripts/newfault_detector.py``: ``trainable = valid & ~(held_out_mask(fold) | held_out_mask(
eval_fold))``).  The union picks and measures on 0 and 1.  This script makes that visible instead of
leaving it in a reader's head, and it quantifies it with a difference-in-differences design so the
number cannot be dismissed as "different folds are just harder".

THE DESIGN
----------
Folds are not equally difficult, so comparing one member's fold 0 with its fold 2 proves nothing.
What controls for fold difficulty is the *difference of differences*:

    D(m) = mean_{f in 0,1} s(m, f)  -  mean_{f in 2,3} s(m, f)

    clean(m) = D(m) for a member whose holdout is folds 2/3      (in-sample on 0/1, not on 2/3)
    control  = mean of D(m) over members whose holdout is 0/1    (the opposite arrangement)

Every member is scored at the SAME policy on the SAME field (the committed ``prob_raw.tif``), so the
only thing that varies between rows is which geography the member saw during fitting.  Under the
null "fold discipline does not matter" the two groups have the same expected D.  A positive gap is
the in-sample signature.

WHAT IT IS NOT.  This is not a claim that a member is bad: a member that memorised folds 0/1 may
still be a fine detector on the private test set, which is a different population of faults rather
than different geography.  What it invalidates is the *measurement* - and therefore any policy
chosen on it.
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
from src.metrics import DEFAULT_R_PIXELS, GtContext, score_within_mask  # noqa: E402
from src.submission_io import sha256_file  # noqa: E402
from src.submission_optim import dilate_mask, floor_sharpen  # noqa: E402

# The 0-based folds the union selects and measures on.  Written as a constant because every number
# in the file is relative to it, and a reader must be able to see what it was.
UNION_SELECTION_FOLD = 0
UNION_MEASUREMENT_FOLD = 1
COMMON_T0 = 0.5          # the common policy every member is scored at (no dilation)


def read_members(root: Path):
    """Every committed member that carries its own report, with the folds it held out."""
    out = []
    for rep in sorted((root / "data/evidence/newfault").glob("*/report.json")):
        r = json.loads(rep.read_text())
        field = rep.parent / "prob_raw.tif"
        if not field.exists():
            continue
        out.append(dict(name=rep.parent.name, field=field, report=r,
                        held_out_fold=r.get("held_out_fold"),
                        measurement_fold=r.get("measurement_fold"),
                        sha256=sha256_file(field)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t0", type=float, default=COMMON_T0)
    ap.add_argument("--root", default=None,
                    help="repository root holding data/ (default: this script's parent repo)")
    ap.add_argument("--out", default="data/evidence/fold_discipline.json")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve() if a.root else ROOT
    with rasterio.open(root / a.labels) as s:
        lab = s.read(1)
        nodata = s.nodata
    valid = np.ones(lab.shape, bool) if nodata is None else (lab != nodata)
    fault = (lab == 1) & valid
    with rasterio.open(root / a.template) as s:
        template = s.read(1)
    footprint = np.isfinite(template)
    with rasterio.open(root / a.proxy) as s:
        proxy = (s.read(1) == 2) & footprint
    H, W = fault.shape
    shape_hw = (H, W)

    table = block_table(shape_hw, a.block_px, valid=footprint, labels=fault)
    fold_of = assign_folds(table, a.folds, a.seed, "balanced")
    masks = {f: scored_mask(shape_hw, a.block_px, fold_of, f) for f in range(a.folds)}
    ctx = GtContext(proxy, DEFAULT_R_PIXELS)

    union_folds = (UNION_SELECTION_FOLD, UNION_MEASUREMENT_FOLD)
    rows = []
    for m in read_members(root):
        with rasterio.open(m["field"]) as s:
            field = np.nan_to_num(s.read(1).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        q = floor_sharpen(field, t0=float(a.t0), hard=True)
        q = (q > 0).astype(np.float32)
        credit = ctx.credit_vector(q)
        scopes = {f"fold{f}": round(score_within_mask(q, ctx, masks[f], credit=credit)["dti"] or 0.0, 4)
                  for f in range(a.folds)}
        union_side = (scopes["fold0"] + scopes["fold1"]) / 2.0
        own = (m["held_out_fold"], m["measurement_fold"])
        own_side = float(np.mean([scopes[f"fold{f}"] for f in own if f in range(a.folds)]))
        # THE CONTROLLED CONTRAST: what the union's two folds score MINUS what the rest of the
        # partition scores.  It is defined for every member, which is what makes the control group
        # possible - "union folds minus own folds" would be identically zero for a member whose
        # holdout already IS the union's folds, i.e. exactly the members that have to serve as the
        # control for fold difficulty.
        other = [f for f in range(a.folds) if f not in union_folds]
        other_side = float(np.mean([scopes[f"fold{f}"] for f in other])) if other else None
        cross = None if other_side is None else round(union_side - other_side, 4)
        rows.append(dict(
            member=m["name"], field=str(m["field"].relative_to(root)), sha256=m["sha256"],
            held_out_folds=list(own),
            in_sample_on_union_scopes=bool(set(own) and not (set(own) & set(union_folds))),
            scopes=scopes, union_scope_mean=round(union_side, 4),
            own_holdout_mean=round(own_side, 4),
            other_folds_mean=(None if other_side is None else round(other_side, 4)),
            union_minus_other_folds=cross))

    # THE ESTIMATE.  For a member whose holdout is folds 2/3 ("treated"), `union_minus_other_folds` is
    # (in-sample geography) - (out-of-sample geography) = fold difficulty + exposure.  For a member
    # whose holdout IS folds 0/1 ("control"), the same quantity is (out-of-sample) - (in-sample) =
    # fold difficulty - exposure.  Averaging the two groups and halving cancels fold difficulty and
    # leaves the exposure term alone.  Without the control group this number would be unidentifiable
    # - folds are not equally hard, and pretending otherwise is the mistake this script exists to
    # avoid.
    treated = [r["union_minus_other_folds"] for r in rows if r["in_sample_on_union_scopes"]
               and r["union_minus_other_folds"] is not None]
    control = [r["union_minus_other_folds"] for r in rows
               if not r["in_sample_on_union_scopes"] and r["union_minus_other_folds"] is not None]
    effect = None
    if treated and control:
        effect = round(float(np.mean(treated) - np.mean(control)) / 2.0, 4)

    clean = [r["member"] for r in rows if not r["in_sample_on_union_scopes"]]
    verdict = (
        "NO COMMITTED MEMBER OF THIS FAMILY TRAINS ON BOTH UNION FOLDS"
        if not treated else
        f"{len(rows) - len(clean)} of {len(rows)} members were fitted on the geography the union "
        f"selects and measures on ({', '.join(r['member'] for r in rows if r['member'] not in clean)}); "
        f"every number the union reports on those folds is an in-sample number for them. Clean "
        f"members: {', '.join(clean) or 'none'}"
    )
    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/audit_fold_discipline.py",
        question=("Is the geography the union uses to SELECT and MEASURE a policy held out by every "
                  "member it unions?"),
        union_selection_fold=UNION_SELECTION_FOLD, union_measurement_fold=UNION_MEASUREMENT_FOLD,
        common_policy=dict(t0=float(a.t0), dilate=0,
                           why=("every member is scored at one common floor on its own committed "
                                "field, so the only variation between rows is training geography")),
        population="proxy (SGMC code 2), the new-fault surrogate",
        partition=dict(block_px=a.block_px, folds=a.folds, seed=a.seed, mode="balanced",
                       n_blocks=int(table["block"].max()) + 1 if "block" in table else None),
        members=rows,
        difference_in_differences=dict(
            treated_mean=(round(float(np.mean(treated)), 4) if treated else None),
            control_mean=(round(float(np.mean(control)), 4) if control else None),
            exposure_effect=effect,
            design=("mean over treated members of [score on the union's folds - score on the rest] "
                    "minus the same mean over controls, halved.  Halving is only exact if the "
                    "in-sample boost is the same kind of boost on both fold sets (B_01 = B_23); "
                    "the two groups are built so the fold-difficulty term cancels exactly, and the "
                    "residual is exposure - with that symmetry assumption as its only gap, stated "
                    "here rather than hidden.  A harsh COMMON floor inflates the magnitude: the "
                    "honest side of the contrast can be near zero"),
            reading=("positive effect = a member trained on the union's folds scores higher there "
                     "than a member for which those folds are the holdout, after removing the "
                     "difference in fold difficulty")),
        verdict=verdict,
        consequence=("A policy selected on a scope that is in-sample for part of the member pool is "
                     "selected on a mixture the sweep cannot claim. The clean pool is the set of "
                     "members whose holdout IS the union's scope; see "
                     "data/evidence/combined_clean/report.json."),
        caveat=("In-sample-ness does not make a member a bad detector - the private scored set is a "
                "different population of faults, not different geography. It makes the MEASUREMENT "
                "optimistic, and therefore any policy chosen with it."),
    )
    dest = root / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[fold-discipline] {verdict}")
    for r in rows:
        print(f"  {r['member']:<10} holdout={tuple(r['held_out_folds'])} "
              f"union-scope mean={r['union_scope_mean']:.4f} own-folds mean={r['own_holdout_mean']:.4f} "
              f"union-minus-other={r['union_minus_other_folds']:+.4f}"
              + ("   <-- IN-SAMPLE on the union's folds" if r["in_sample_on_union_scopes"] else ""))
    if effect is not None:
        print(f"  in-sample exposure (difference-in-differences / 2): {effect:+.4f}")
    print(f"[fold-discipline] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
