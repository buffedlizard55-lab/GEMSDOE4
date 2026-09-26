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


SELECTION_KEY = "proxy_dti_selection"
"""The ONE key the sweep is allowed to pick a policy with: the new-fault-population DTI on the
SELECTION fold's blocks.  It is a module constant so a reviewer can grep for it, and so the test
suite can assert that the whole-grid number (`proxy_dti`) never becomes the ranking key again."""


def select_winner(rows, key: str = SELECTION_KEY):
    """PRE-REGISTERED selection rule: argmax of `key`, ties broken by narrower dilation then by
    lower agreement.

    THE DEFECT THIS REPLACES (found and fixed 2026-09-26, session 34).  This line used to read
    `max(eligible, key=lambda r: (r["proxy_dti"], ...))` - the WHOLE-GRID proxy DTI - while the
    docstring, `docs/FIELD_SELECTION_RULE.md` and the report's own `selection.scope` said the sweep
    selected on the SELECTION fold's blocks.  The whole grid is not a noisier version of the
    selection fold: `newfault_detector.py` trains each member on everything except folds 0 and 1, so
    folds 2 and 3 are IN-SAMPLE for three of the five members and the whole-grid number spends most
    of its truth mass there.  Selection on it is selection on geography the members have seen.

    Nothing here reads `proxy_dti` (whole grid) or `proxy_dti_measurement` (the held-out fold).
    """
    if not rows:
        raise SystemExit("no eligible candidate: every combination exceeded the support window")
    for r in rows:
        if r.get(key) is None:
            raise SystemExit(f"candidate {r} has no {key}: a fold with no truth pixels cannot "
                             f"select a policy, and treating the missing value as 0.0 would.")
    return max(rows, key=lambda r: (r[key], -r["dilate"], -r["vote"]))


def _fmt(v) -> str:
    """Format a possibly-undefined DTI.  A scope with no truth pixels has no index (`None`), and
    printing it as 0.0 would look like a measured failure instead of an unmeasurable scope."""
    return "n/a" if v is None else f"{v:.4f}"


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


def _member_holdout(member: dict):
    """The folds a member's own report says it trained WITHOUT, as a set of ints (possibly empty).

    Read from the member's report, never assumed: a member whose runner leaves no holdout record
    (the deep ensemble trains over the whole grid) returns an empty set, which is the honest
    answer - it held out nothing - and never a guess.
    """
    prov = member.get("provenance") or {}
    held = set()
    for key in ("held_out_fold", "measurement_fold"):
        v = prov.get(key)
        if isinstance(v, int) or (isinstance(v, str) and v.strip().isdigit()):
            held.add(int(v))
    return held


def _fold_discipline(members, sel_fold: int, eval_fold: int) -> dict:
    """Per-member answer to: did this member train on the geography the union uses?

    The union selects on fold ``sel_fold`` and measures on fold ``eval_fold`` and DISCLOSES
    ``sel_fold+eval_fold``.  A member is "clean" for a scope only if its own report says both of
    that scope's folds were held out; anything else is listed as in-sample.  Members with no
    holdout protocol (whole-grid training) are never counted as clean.
    """
    rows, dirty = [], []
    for m in members:
        held = _member_holdout(m)
        needs_sel, needs_eval = {int(sel_fold)}, {int(eval_fold)}
        clean_sel = needs_sel <= held
        clean_eval = needs_eval <= held
        clean_pool = (needs_sel | needs_eval) <= held
        if not clean_pool:
            dirty.append(m["name"])
        rows.append(dict(
            name=m["name"], held_out_folds=sorted(held),
            protocol=("no fold protocol (whole-grid training)" if not held
                      else (f"trained on everything except folds {sorted(held)}")),
            clean_on_selection_fold=bool(clean_sel), clean_on_measurement_fold=bool(clean_eval),
            clean_on_the_pooled_pool=bool(clean_pool)))
    clean = [r["name"] for r in rows if r["clean_on_the_pooled_pool"]]
    return dict(
        selection_fold=int(sel_fold), measurement_fold=int(eval_fold),
        per_member=rows, clean_pool_members=clean, in_sample_members=dirty,
        verdict=("every member held out both folds" if not dirty else
                 f"{len(dirty)} of {len(rows)} members were trained on part of the geography the "
                 f"union selects or measures on: {', '.join(dirty)}"),
        consequence=("A policy selected on a scope that is in-sample for part of the pool is "
                     "selected on a mixture the sweep cannot claim; the clean pool's own run is "
                     "data/evidence/combined_clean/report.json.  Re-running the offending members "
                     "with --fold 0 --eval-fold 1 is queued in SUGGESTIONS.md."),
        audit=f"python scripts/audit_fold_discipline.py   # -> data/evidence/fold_discipline.json",
    )


def _in_sample_note(members, sel_fold: int, eval_fold: int, which: str) -> str:
    """One sentence naming the members that are in-sample on the scope being described."""
    def bad(folds):
        return [m["name"] for m in members
                if not (set(folds) <= _member_holdout(m))]

    if which == "selection":
        who, folds = bad([sel_fold]), [sel_fold]
    elif which == "measurement":
        who, folds = bad([eval_fold]), [eval_fold]
    else:
        who, folds = bad([sel_fold, eval_fold]), [sel_fold, eval_fold]
    if not who:
        return (f"Every member held out fold{'s' if len(folds) > 1 else ''} "
                f"{', '.join(str(f) for f in folds)}.")
    return (f"IN-SAMPLE for {len(who)} of {len(members)} members "
            f"({', '.join(who)}): their training geography includes fold"
            f"{'s' if len(folds) > 1 else ''} "
            f"{', '.join(str(f) for f in folds)}. See selection.fold_discipline.")


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
    ap.add_argument("--floors", type=int, default=24)
    ap.add_argument("--dilates", default="0,1")
    # Session 34: k = 1 and 2 were the only votes ever swept, and the k = 3 family is the one that
    # generalises best out of sample (see STATUS.md §Session 34).  A search that cannot express the
    # winning hypothesis is not a search of that hypothesis space, so the default now spans k = 1..5
    # and values above the member count are refused by name rather than silently emitting nothing.
    ap.add_argument("--votes", default="1,2,3,4,5",
                    help="comma-separated k-of-n agreement thresholds (k <= number of members)")
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

    # THE SELECTION SCOPE IS THE SELECTION FOLD, NOT THE WHOLE GRID (fixed 2026-09-26).
    # Until session 34 every row was scored with `ctx[pop].score(field)`, i.e. over the WHOLE grid,
    # while the docstring, the report's `selection.scope` and `ranking_key` all said "the SELECTION
    # fold's blocks".  The two are not the same measurement, and the difference is not cosmetic:
    # `scripts/newfault_detector.py:336-341` trains each NFF member on everything EXCEPT folds 0
    # and 1, so folds 2 and 3 are IN-SAMPLE for three of the five members, and a whole-grid score
    # spends 58 % of its truth mass on geography the members have already seen.  Selecting a policy
    # on that number is selection on a mixture the sweep cannot honestly claim.  Every candidate is
    # now scored on the selection fold (the key that selects), the measurement fold, the two pooled
    # (17 blocks, the honest pool) and the whole grid (reported for continuity, never used to pick).
    pool_mask = select_mask | eval_mask
    sel_ctx = {
        "selection": (select_mask, ctx["proxy"]),
        "measurement": (eval_mask, ctx["proxy"]),
        "pooled01": (pool_mask, ctx["proxy"]),
        "whole": (None, ctx["proxy"]),
    }
    rows = []
    for t0 in t0_grid:
        for dil in dilates:
            for k in votes:
                if k > len(members):
                    # k > n would emit nothing at all in real runs, but a *default* grid that
                    # silently contains an impossible vote is a grid that lies about what it
                    # searched: record the refusal instead of a row of zeros.
                    print(f"[combine] skipping k={k}: only {len(members)} members", flush=True)
                    continue
                field = combine(t0, dil, k)
                row = dict(t0=float(t0), dilate=int(dil), vote=int(k),
                           emitted_px=int((field > 0).sum()))
                credit = ctx["proxy"].credit_vector(field)
                for scope, (mask, c) in sel_ctx.items():
                    if mask is None:
                        row[f"proxy_dti_{scope}"] = c.score(field)
                    else:
                        s = score_within_mask(field, c, mask, credit=credit)
                        row[f"proxy_dti_{scope}"] = s["dti"]
                        row[f"proxy_tp_{scope}"] = s["TP_w"]
                        row[f"proxy_fp_{scope}"] = s["FP_w"]
                        row[f"proxy_n_gt_{scope}"] = s["n_gt"]
                # back-compatible names: `proxy_dti`/`catalogue_dti` stay the WHOLE-GRID numbers
                # because build_site.py and build_submission_payload.py render them as such.  They
                # are context.  The pick below reads `proxy_dti_selection` and nothing else.
                row["proxy_dti"] = row["proxy_dti_whole"]
                row["catalogue_dti"] = ctx["catalogue"].score(field)
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
                print(f"[combine] t0={t0:.6g} dil={dil} k={k} "
                      f"sel={_fmt(row['proxy_dti_selection'])} "
                      f"mea={_fmt(row['proxy_dti_measurement'])} "
                      f"pooled={_fmt(row['proxy_dti_pooled01'])} "
                      f"whole={row['proxy_dti']:.4f} catalogue={row['catalogue_dti']:.4f} "
                      f"px={row['emitted_px']:,}", flush=True)

    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise SystemExit("no eligible candidate: every combination exceeded the support window")
    winner = select_winner(eligible)
    print(f"[combine] selected t0={winner['t0']} dilate={winner['dilate']} k={winner['vote']} "
          f"selection_fold_proxy_dti={winner['proxy_dti_selection']:.4f} "
          f"(measurement fold {_fmt(winner['proxy_dti_measurement'])}, "
          f"pooled folds 0+1 {_fmt(winner['proxy_dti_pooled01'])}, "
          f"whole grid {winner['proxy_dti']:.4f})", flush=True)

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
    # THE SIDECAR MUST FOLLOW THE BYTES (session 34).  `scripts/package_submission.py`,
    # `tests/test_package_submission.py` and `scripts/check_submission_readiness.py` all read
    # `submission.sha256` as the record of what was written, so leaving a previous run's hash in
    # place publishes a hash that belongs to bytes no longer on disk - a defect this repository has
    # already hit once (session 33: a stale page describing a superseded artifact).  The writer is
    # the only place that knows the hash, so the writer updates the file.
    sidecar = sub_path.with_name("submission.sha256")
    sidecar.write_text(f"{sha256_file(sub_path)}  {sub_path}\n")

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
                       ranking_key=("proxy_dti_selection desc (new-fault-population DTI on the "
                                    "SELECTION fold's blocks), then narrower dilation, then lower k"),
                       selection_key="proxy_dti_selection",
                       # COMPUTED, NOT ASSERTED (session 34).  The earlier text of this block said
                       # folds 0+1 were "the two folds the NFF members never trained on".  That is
                       # true of the members run with --fold 0 --eval-fold 1 and FALSE for the two
                       # members run with --fold 2 --eval-fold 3: `newfault_detector.py` trains on
                       # `valid & ~(held_out_mask(fold) | held_out_mask(eval_fold))`, so for those
                       # members folds 0 and 1 are TRAINING geography.  The member list below is
                       # read from each member's own report.json, so the disclosure cannot drift
                       # from the bytes it describes.
                       fold_discipline=_fold_discipline(members, args.fold, args.eval_fold),
                       scopes_explained={
                           "proxy_dti_selection": (
                               f"DTI on the blocks of fold {args.fold} - the ONLY key that selects. "
                               + _in_sample_note(members, args.fold, args.eval_fold, "selection")),
                           "proxy_dti_measurement": (
                               f"DTI on the blocks of fold {args.eval_fold} - the number to quote, "
                               "never a selection key. "
                               + _in_sample_note(members, args.fold, args.eval_fold, "measurement")),
                           "proxy_dti_pooled01": (
                               f"DTI on the pooled disjoint blocks of folds {args.fold}+"
                               f"{args.eval_fold}. "
                               + _in_sample_note(members, args.fold, args.eval_fold, "either")
                               + " Fold "
                               f"{args.fold} is also a selection fold, so the pooled number is not "
                               "pristine even for a member that held both folds out."),
                           "proxy_dti_whole": (
                               "whole-grid DTI - REPORTED, NEVER USED TO SELECT: it mixes folds the "
                               "members held out with folds they trained on, for every member that "
                               "has a fold protocol at all, so it cannot select anything.")},
                       k_votes_swept=[int(x) for x in str(args.votes).split(",") if x != ""],
                       n_members=len(members),
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
