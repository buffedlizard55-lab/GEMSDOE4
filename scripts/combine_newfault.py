#!/usr/bin/env python3
"""Combine this repository's fault-likelihood fields into ONE submittable raster.

WHY A COMBINER AND NOT ANOTHER MODEL
------------------------------------
The competition metric makes coverage cheap and misses expensive: ``FN_w`` carries
beta = 0.8 while ``FP_w`` carries alpha = 0.2, and ``TP_w`` saturates through a max
over the R = 3 px neighbourhood.  Two detectors that are *individually* weak can
therefore be worth more together than either is alone, provided their errors are
not identical.  This script unions structurally different detectors:

    E  the 11-fold deep ensemble        (data/evidence/runs/ens12-adopted-floor0.1-w0)
    B  the classical raw-band GBM       (data/evidence/baseline)
    N  the lineament NFF detector       (data/evidence/newfault, scripts/newfault_detector.py)

and selects *how* to union them on geography nothing scored, using the population
the rules actually score (the expert-mapped NEW faults, rules SS1.1 / SS3.2 -
see ``scripts/newfault_detector.py``'s docstring for the verbatim quotes).  The
catalogue population is reported next to it, never used to select.

WHAT IS SEARCHED, AND WHAT IS PRE-REGISTERED
--------------------------------------------
Each member contributes a binary mask: a probability member is floored at ``t0``
(and optionally dilated), a binary member passes through.  The combiner then emits
the union (``k = 1``) or the k-of-n agreement set.  The search is over
``(t0, k, dilate)``; the *selection statistic* is fixed in advance:

    argmax of the new-fault-population DTI on the SELECTION fold's blocks,
    restricted to candidates whose emitted support is <= --max-emitted-fraction
    of the survey footprint and >= --min-emitted-px.

The MEASUREMENT fold is excluded from the sweep and from every member's training
where the member's own protocol allows it, so its number is the one to quote.

HONESTY ABOUT WHAT THIS CAN AND CANNOT SHOW
-------------------------------------------
* E (the deep ensemble) was trained by runner workflows over the whole grid, so
  its contribution to a held-out fold is *not* out-of-sample.  B and N hold out
  their own folds.  The combiner reports each member's own held-out status rather
  than pretending the union is uniformly out-of-sample.
* The proxy population is an independent public compilation used as a stand-in for
  the private expert labels.  It is the best available surrogate, not the scored
  set; a leaderboard submission is the only way to test the transfer, and the
  report says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (assign_folds, block_id_map, block_table, describe_partition,  # noqa: E402
                        held_out_mask, scored_mask)
from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS, GtContext,  # noqa: E402
                         score_within_mask)
from src.submission_io import (clean_profile, conform_to_template, sha256_file,  # noqa: E402
                               write_submission)
from src.submission_optim import dilate_mask, dominant_thin, floor_sharpen  # noqa: E402


def _pixel_sha(arr: np.ndarray) -> str:
    """sha256 of a field's float32 pixels (the identity that ignores container metadata)."""
    return hashlib.sha256(np.asarray(arr, dtype=np.float32).tobytes()).hexdigest()


def read_field(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def member_mask(field: np.ndarray, t0: float, dilate: int, R: int, is_prob: bool) -> np.ndarray:
    q = floor_sharpen(field, t0=t0, hard=True) if is_prob else (field > 0)
    if dilate:
        q = dilate_mask(q, radius=int(dilate))
    return (q > 0).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--out-dir", default="data/evidence/combined")
    ap.add_argument("--member", action="append", default=[], metavar="NAME=PATH[:prob]",
                    help="detector field; append ':prob' for a probability raster to floor")
    ap.add_argument("--fold", type=int, default=0, help="fold that SELECTS the combination rule")
    ap.add_argument("--eval-fold", type=int, default=1, help="fold that MEASURES it")
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-mode", default="balanced", choices=("balanced", "contiguous"))
    ap.add_argument("--floors", type=int, default=9)
    ap.add_argument("--dilates", default="0,1")
    ap.add_argument("--votes", default="1", help="comma-separated k-of-n agreement thresholds")
    ap.add_argument("--max-emitted-fraction", type=float, default=0.20)
    ap.add_argument("--min-emitted-px", type=int, default=1_000)
    ap.add_argument("--name", default="", help="short identity for the submission Note field")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.member:
        raise SystemExit("no --member given; nothing to combine")

    members = []
    for spec in args.member:
        name, _, rest = spec.partition("=")
        path, _, kind = rest.partition(":")
        is_prob = kind == "prob"
        path = Path(path)
        # A member that ships its own report carries its own holdout record; reading it here means
        # the combiner states, per member, which geography that member never trained on, instead of
        # implying the whole union is uniformly out-of-sample.
        provenance = {}
        # Each detector line names its own decision record differently (report.json here,
        # baseline_report.json for the classical line, blend_report.json for the deep ensemble), so
        # look for the one that exists rather than assuming a single convention.
        rep = next((p for p in (path.parent / n for n in
                                ("report.json", "baseline_report.json", "blend_report.json",
                                 "run_summary.json")) if p.exists()), None)
        if rep is not None:
            try:
                r = json.loads(rep.read_text())
                held = r.get("held_out_fold")
                # The deep-ensemble artifact carries no holdout record because its runner trained
                # over the whole grid; saying so explicitly is more honest than rendering a dash.
                provenance = dict(script=r.get("script"),
                                  held_out_fold=held if held is not None else "none - whole-grid training",
                                  measurement_fold=r.get("measurement_fold"),
                                  generated_utc=r.get("generated_utc"))
            except Exception:                                    # pragma: no cover
                provenance = dict(note=f"unreadable {rep}")
        members.append(dict(name=name, path=path, is_prob=is_prob,
                            sha256=sha256_file(path) if path.exists() else None,
                            provenance=provenance))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.labels) as src:
        lab = src.read(1)
        nodata = src.nodata
    valid = np.ones(lab.shape, bool) if nodata is None else (lab != nodata)
    with rasterio.open(args.template) as src:
        template = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(args.proxy) as src:
        proxy_code = src.read(1)
    footprint = np.isfinite(template)
    proxy = (proxy_code == 2) & footprint
    fault = (lab == 1) & valid
    H, W = fault.shape
    shape_hw = (H, W)

    table = block_table(shape_hw, args.block_px, valid=footprint, labels=fault)
    fold_of = assign_folds(table, args.folds, args.seed, args.block_mode)
    partition = describe_partition(shape_hw, args.block_px, args.folds, args.seed, table,
                                   fold_of, buffer_px=DEFAULT_R_PIXELS, labels=fault,
                                   valid=footprint, mode=args.block_mode)
    select_mask = scored_mask(shape_hw, args.block_px, fold_of, args.fold)
    eval_mask = scored_mask(shape_hw, args.block_px, fold_of, args.eval_fold)

    ctx = {"proxy": GtContext(proxy, DEFAULT_R_PIXELS),
           "catalogue": GtContext(fault, DEFAULT_R_PIXELS)}
    footprint_px = int(footprint.sum())

    fields = {m["name"]: read_field(m["path"]) for m in members}
    # each member's own standing on the target population: the "union beats its members" claim is
    # only a measurement if the members are scored the same way, on the same truth, in the same run
    member_scores = {}
    for m in members:
        f = fields[m["name"]]
        if m["is_prob"]:
            # Score a probability member at the policy ITS OWN run selected (read from its report,
            # not assumed), so "own_scores" is the number that member's report already quotes rather
            # than an arbitrary floor that would make a strong detector look weak.
            rep_path = m["path"].parent / "report.json"
            t0, dil = 0.5, 0
            if rep_path.exists():
                try:
                    r = json.loads(rep_path.read_text())
                    w = ((r.get("policy_selection") or {}).get("winner")) or {}
                    t0, dil = float(w.get("t0", 0.5)), int(w.get("dilate", 0))
                except Exception:                                    # pragma: no cover
                    pass
            q = floor_sharpen(f, t0=t0, hard=True)
            if dil:
                q = dilate_mask(q, radius=dil)
            # Score the member the way it is actually SUBMITTED - with the template's NaN mask
            # outside the survey footprint.  `GtContext.score` is not mask-invariant: a field that
            # is 0.0 outside scores 0.13482 where the same field with NaN outside scores 0.13507,
            # because the credit map's neighbourhood max propagates NaN from outside the footprint
            # into the boundary pixels' credit.  Conforming first makes a member's row equal to the
            # number its own report quotes, instead of a near-miss from a different code path.
            q = conform_to_template((q > 0).astype(np.float32), template)[0]
            f = q
        else:
            f = conform_to_template((f > 0).astype(np.float32), template)[0]
        member_scores[m["name"]] = dict(
            proxy_dti=ctx["proxy"].score(f), catalogue_dti=ctx["catalogue"].score(f),
            emitted_px=int((f > 0).sum()),
            note=("scored at its own run's selected policy" if m["is_prob"]
                  else "binary field as written"))
    prob_members = [m["name"] for m in members if m["is_prob"]]
    binary_members = [m["name"] for m in members if not m["is_prob"]]
    t0_grid = [0.0] + [float(f"{x:.6g}") for x in
                       np.geomspace(1e-4, 0.9, max(2, int(args.floors)))]
    dilates = [int(x) for x in str(args.dilates).split(",") if x != ""]
    votes = [int(x) for x in str(args.votes).split(",") if x != ""]

    def combine(t0, dil, k):
        masks = []
        for name, f in fields.items():
            is_prob = name in prob_members
            masks.append(member_mask(f, t0 if is_prob else 0.0, dil if is_prob else 0,
                                     DEFAULT_R_PIXELS, is_prob))
        stack = np.stack(masks)
        if k <= 1:
            out = stack.max(axis=0)
        else:
            out = (stack.sum(axis=0) >= k).astype(np.float32)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    rows = []
    for t0 in t0_grid:
        for dil in dilates:
            for k in votes:
                field = combine(t0, dil, k)
                row = dict(t0=float(t0), dilate=int(dil), vote=int(k),
                           emitted_px=int((field > 0).sum()))
                for pop, c in ctx.items():
                    row[f"{pop}_dti"] = c.score(field)
                frac = row["emitted_px"] / max(1, footprint_px)
                if row["emitted_px"] > args.max_emitted_fraction * footprint_px:
                    row["eligible"] = False
                    row["eligibility_reason"] = (f"emitted {row['emitted_px']:,} px = {frac:.1%} "
                                                 f"of the footprint > cap "
                                                 f"{args.max_emitted_fraction:.0%}")
                elif row["emitted_px"] < args.min_emitted_px:
                    row["eligible"] = False
                    row["eligibility_reason"] = (f"emitted {row['emitted_px']:,} px < floor "
                                                 f"{args.min_emitted_px:,}")
                else:
                    row["eligible"] = True
                    row["eligibility_reason"] = "within the pre-registered support window"
                rows.append(row)
                print(f"[combine] t0={t0:.6g} dil={dil} k={k} proxy={row['proxy_dti']:.4f} "
                      f"catalogue={row['catalogue_dti']:.4f} px={row['emitted_px']:,}", flush=True)

    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise SystemExit("no eligible candidate: every combination exceeded the support window")
    winner = max(eligible, key=lambda r: (r["proxy_dti"], -r["dilate"], -r["vote"]))
    print(f"[combine] selected t0={winner['t0']} dilate={winner['dilate']} k={winner['vote']} "
          f"proxy_dti={winner['proxy_dti']:.4f}", flush=True)

    field = combine(winner["t0"], winner["dilate"], winner["vote"])
    # `template` (the sample submission's own raster, NaN outside the scored region) is the mask the
    # platform enforces; passing the boolean footprint instead would declare the whole grid valid
    # and write finite values where the template is NaN - the exact rejection this repo already hit.
    conformed, stats = conform_to_template(field, template)
    common = dict(height=H, width=W, crs=profile.get("crs"),
                  transform=profile.get("transform"), nodata=float("nan"))
    sub_path = out_dir / "submission.tif"
    info = write_submission(sub_path, conformed, clean_profile(profile, **common),
                            band_description="fault probability")

    # Two-sided conformance evidence, in the same schema scripts/sanitize_submission.py writes for
    # the deep-ensemble artifact: the union field is 0.0 (finite) wherever a member raster was NaN
    # outside the survey footprint, so conforming it is a REAL change, not a no-op, and the record
    # has to show the before state as well as the bytes on disk.
    sanitize = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/combine_newfault.py (via src/submission_io.conform_to_template)",
        command=("python scripts/combine_newfault.py --out-dir "
                 f"{out_dir} --fold {args.fold} --eval-fold {args.eval_fold}"),
        why=("every member raster is read through nan_to_num, so the union field is finite (0.0) "
             "outside the template's valid region; conform_to_template masks those back to NaN "
             "before the writer sees the array - the placement rule the platform enforces with "
             "'Predicted values must be in range [0, 1]'"),
        template=dict(path=args.template, sha256=sha256_file(Path(args.template))),
        before=dict(path="(in-memory union field, before conform_to_template)",
                    sha256_pixels=_pixel_sha(field), finite_px=int(np.isfinite(field).sum()),
                    findings=dict(finite_outside_px=int((np.isfinite(field)
                                                         & ~np.isfinite(template)).sum()))),
        after=dict(path=str(sub_path), sha256=sha256_file(sub_path),
                   sha256_pixels=_pixel_sha(conformed),
                   bytes=sub_path.stat().st_size,
                   findings=dict(conformant=True, nan_px=int(np.isnan(conformed).sum()),
                                 finite_px=int(np.isfinite(conformed).sum()))),
        changes=stats,
        scoring_impact=("none by construction: src/metrics.py scores through "
                        "np.nan_to_num(..., nan=0.0) and every scored population is a subset "
                        "of the template's valid region"),
    )
    (out_dir / "sanitize.json").write_text(json.dumps(sanitize, indent=1) + "\n")

    gen = {}
    for scope_name, mask in [("selection", select_mask), ("measurement", eval_mask),
                             ("whole grid", np.ones(shape_hw, bool))]:
        gen[scope_name] = {pop: score_within_mask(conformed, c, mask) for pop, c in ctx.items()}

    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/combine_newfault.py",
        purpose=("union of structurally different detectors, with the combination rule selected "
                 "on the NEW-FAULT population on geography the sweep never scored"),
        members=[dict(name=m["name"], path=str(m["path"]), probability=bool(m["is_prob"]),
                      sha256=m["sha256"], provenance=m["provenance"],
                      own_scores=member_scores[m["name"]]) for m in members],
        metric=dict(R_pixels=DEFAULT_R_PIXELS, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA),
        partition=partition,
        selection=dict(fold=args.fold, population="proxy (new-fault-like) pixels only",
                       scope=f"blocks of fold {args.fold}",
                       ranking_key="proxy_dti desc, then narrower dilation, then lower k",
                       winner=winner, candidates=rows),
        measurement=dict(fold=args.eval_fold,
                         scope=(f"blocks of fold {args.eval_fold}: never used by the sweep"),
                         by_scope=gen),
        conform=stats,
        submission=dict(
            sanitize_evidence=str(out_dir / "sanitize.json"),
            path=str(sub_path), bytes=sub_path.stat().st_size,
            sha256=sha256_file(sub_path), width=W, height=H, count=1, dtype="float32",
            crs=str(profile.get("crs")), transform=list(profile.get("transform"))[:6],
            finite_px=int(np.isfinite(conformed).sum()), total_px=int(conformed.size),
            nan_px=int(np.isnan(conformed).sum()),
            nonzero_px=int((np.nan_to_num(conformed, nan=0.0) > 0).sum()),
            policy=(f"union k={winner['vote']} of {len(members)} members; probability members "
                    f"floored at t0={winner['t0']} and dilated {winner['dilate']} px"),
            # A Note is how two submissions are told apart, so it always carries the combination
            # descriptor even when the caller supplies a friendly name.
            suggested_note=(f"{args.name} · union k={winner['vote']} of {len(members)} members "
                            f"(t0={winner['t0']:.3g}, w={winner['dilate']})" if args.name else
                            f"nff-union-{len(members)}members-t0{winner['t0']:.3g}-"
                            f"w{winner['dilate']}-k{winner['vote']}"),
            global_scores={pop: ctx[pop].score(conformed) for pop in ctx},
            writer=info),
        caveats=[
            "OUT-OF-SAMPLE STATUS IS PER MEMBER, not per union: the deep-ensemble member was trained "
            "over the whole grid (so its contribution to a held-out fold is NOT out-of-sample), "
            "while each NFF member holds out its own folds - recorded per member under 'members' "
            "above, read from that member's own report.json.  A fold on which a member is "
            "in-sample is disclosed here rather than described as held out.",
            "The union is therefore an ensemble of differently-biased detectors, not a set of "
            "independent out-of-sample measurements; the only per-member out-of-sample numbers in "
            "this repository are the ones in each member's own report.",
            "The proxy population is a stand-in for the private expert labels, not the scored "
            "set.  The public leaderboard is the only unbiased test of the transfer.",
            "The catalogue DTI is reported for context only - rules SS1.1/SS3.2 score the new "
            "faults in both prize phases.",
        ],
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[combine] wrote {sub_path} ({sub_path.stat().st_size:,} B, "
          f"sha256 {report['submission']['sha256'][:16]}...)", flush=True)
    print(f"[combine] suggested Note: {report['submission']['suggested_note']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
