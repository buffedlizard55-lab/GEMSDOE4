#!/usr/bin/env python3
"""New-Fault-First (NFF) detector: a CPU-only, torch-free fault-likelihood field.

WHY THIS EXISTS - the scored population is NEW faults, not the catalogue
-----------------------------------------------------------------------
The official rules fix what is scored, verbatim (``data/evidence/rules_quotes.json``):

    SS1.1  "In Phase 1, submissions will be evaluated against a privately withheld
            subset of the original new fault dataset compiled by expert reviewers."
    SS1.1  "Submissions will be reevaluated against the full, revised new fault
            dataset using the same distance-weighted Tversky index."
    SS3.2  "Second-round prize rankings will be determined by running the selected
            final submissions against the complete updated test set created by
            expert review."

Both prize phases therefore score the *expert-mapped new faults*.  The USGS
catalogue in ``data/labels.tif`` is the training set, never the test set.  This
script is built around that fact:

1. **Supervision from an independent fault compilation.**  The catalogue teaches
   the model the faults geologists already mapped - i.e. the faults with the most
   obvious geophysical expression.  Adding an independent public compilation
   (``data/evidence/proxy/proxy_catalogue.tif``, rasterised from the SGMC
   statewide geology) gives the model examples of faults the catalogue does NOT
   contain, which is the population the metric rewards.  Three supervision modes
   are available so *member diversity* can come from the training target rather
   than from random seeds alone (session-31 finding: same features + same truth
   ⇒ correlated errors; extra same-target members add FP mass faster than
   coverage):

   * ``--supervision union`` (default) — catalogue ∪ proxy (code 1 near-label
     + code 2 proxy-only).  The production default.
   * ``--supervision proxy_only`` — ONLY the proxy-only (code 2) pixels.  The
     member never sees a catalogue fault as a positive, so its errors are
     structurally anti-correlated with catalogue-trained detectors.  This is
     the GEMSDOE4 unique angle against seed-only ensembles.
   * ``--supervision catalogue`` — catalogue only (the ``--no-proxy-labels``
     ablation, kept as a flag synonym).
2. **Lineament geometry as features** (``src/lineament_features.py``): multi-scale
   ridge-ness, structure-tensor coherence and windowed context, so the classifier
   can see *lines* rather than isolated pixel values.
3. **Selection on the new-fault population, on geography the model never saw.**
   Fold ``--fold`` selects the emission policy by the proxy (new-fault) DTI; fold
   ``--eval-fold`` - excluded from training and never scored during the sweep -
   measures it.  The catalogue DTI is reported for context only; it is not the
   selection signal because the catalogue is not what gets scored.
4. **A pre-registered eligibility window** (support <= 5 % of the footprint,
   >= 1,000 px).  ``FN_w`` carries beta = 0.8 against ``FP_w``'s alpha = 0.2, so an
   unconstrained argmax over the shaping space is "paint the whole footprint",
   which is not a prediction.  Ineligible rows stay in the report with the reason.

Nothing is fitted on the held-out geography, no network access is needed, and the
whole path runs in the same 2 vCPU / ~4 GB sandbox that cannot hold the feature
stack twice.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (assign_folds, block_id_map, block_table, describe_partition,  # noqa: E402
                        held_out_mask, scored_mask)
from src.lineament_features import lineament_features  # noqa: E402
from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS, GtContext,  # noqa: E402
                        kernel_offsets, score_within_mask)
from src.submission_io import (clean_profile, conform_to_template, sha256_file,  # noqa: E402
                               write_submission)
from src.submission_optim import dilate_mask, dominant_thin, floor_sharpen  # noqa: E402

NODATA_FLOOR = -1e30
HALO_PX = 16                 # > 3*sigma of the largest filter kernel; keeps chunk edges exact
CHUNK_ROWS = 256             # interior rows per pass (halo added on both sides)
FLOOR_GRID = 9
DILATE_GRID = (0, 1)
MAX_EMITTED_FRACTION = 0.05
MIN_EMITTED_PX = 1_000
AUDIT_TOLERANCE = 5e-4


# --------------------------------------------------------------------------------- reading
def _nan_nodata(block: np.ndarray) -> np.ndarray:
    out = np.asarray(block, dtype=np.float32).copy()
    out[out < NODATA_FLOOR] = np.nan
    return out


def read_truth(labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(fault mask, valid mask) from labels.tif: code 1 = fault, nodata = outside the survey."""
    with rasterio.open(labels_path) as src:
        a = src.read(1)
        nodata = src.nodata
    valid = np.ones(a.shape, bool) if nodata is None else (a != nodata)
    return (a == 1) & valid, valid


def read_template(path: Path) -> tuple[np.ndarray, dict]:
    """Official footprint mask (finite pixels of sample_submission.tif) and its profile."""
    with rasterio.open(path) as src:
        a = src.read(1)
        prof = src.profile.copy()
    return np.isfinite(a), prof


def read_code_mask(path: Path, code: int) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1) == code


# --------------------------------------------------------------------------------- features
def _row_chunks(H: int, chunk_rows: int = CHUNK_ROWS):
    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        yield r0, r1


def build_feature_rows(feature_path: Path, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix for the pixels where ``wanted`` is True, read in haloed row chunks.

    ``wanted`` is a full-grid boolean mask.  The return is ``(X, idx)`` with ``idx`` the
    flat indices (row-major) of the selected pixels, in ascending order.
    """
    H, W = wanted.shape
    idx = np.flatnonzero(wanted.ravel())
    if idx.size == 0:
        return np.empty((0, 0), np.float32), idx
    rows = (idx // W).astype(np.int64)
    out_cols: list[np.ndarray] = []
    t0 = time.time()
    with rasterio.open(feature_path) as src:
        for r0, r1 in _row_chunks(H):
            # `rows` is non-decreasing, so the chunk's slice is two binary searches: the first
            # selected pixel with row >= r0 up to the first with row >= r1.  Both bounds use
            # side="left"; using "right" on the upper bound re-reads the boundary row and
            # duplicates those pixels, which silently misaligns features against labels.
            lo = int(np.searchsorted(rows, r0, side="left"))
            hi = int(np.searchsorted(rows, r1, side="left"))
            if lo >= hi:
                continue
            y0, y1 = max(0, r0 - HALO_PX), min(H, r1 + HALO_PX)
            block = _nan_nodata(src.read(window=Window(0, y0, W, y1 - y0)))     # (b, h, w)
            Xc, _ = lineament_features(block)
            sub = idx[lo:hi]
            local = (sub // W) - y0
            rows_out = Xc[local * W + (sub % W)]
            out_cols.append(rows_out)
            del block, Xc
    X = np.concatenate(out_cols, axis=0).astype(np.float32)
    return X, idx


def predict_grid(clf, feature_path: Path, footprint: np.ndarray, valid: np.ndarray
                 ) -> tuple[np.ndarray, float]:
    """Full-grid probability field; NaN outside the footprint or where features are missing."""
    H, W = footprint.shape
    prob = np.full((H, W), np.nan, np.float32)
    t0 = time.time()
    with rasterio.open(feature_path) as src:
        for r0, r1 in _row_chunks(H):
            y0, y1 = max(0, r0 - HALO_PX), min(H, r1 + HALO_PX)
            block = _nan_nodata(src.read(window=Window(0, y0, W, y1 - y0)))
            Xc, _ = lineament_features(block)
            interior = np.zeros((r1 - r0, W), bool)
            interior[:] = (valid[r0:r1] & footprint[r0:r1])
            flat = interior.ravel()
            if flat.any():
                sel = np.flatnonzero(flat)
                row = prob[r0:r1].ravel()
                row[sel] = clf.predict_proba(Xc[sel])[:, 1].astype(np.float32)
                prob[r0:r1] = row.reshape(r1 - r0, W)
            del block, Xc
    return prob, round(time.time() - t0, 1)


# --------------------------------------------------------------------------------- sampling
def choose_samples(pos: np.ndarray, trainable: np.ndarray, rng: np.random.Generator,
                   neg_ratio: float, max_negatives: int) -> dict:
    pos = pos & trainable
    neg_pool = trainable & ~pos
    n_pos = int(pos.sum())
    n_pool = int(neg_pool.sum())
    want = min(int(max_negatives), max(1, int(round(neg_ratio * max(1, n_pos)))))
    if want >= n_pool:
        neg = neg_pool
    else:
        flat_idx = np.flatnonzero(neg_pool.ravel())
        pick = np.sort(rng.choice(flat_idx.size, size=want, replace=False))
        neg = np.zeros(neg_pool.size, bool)
        neg[flat_idx[pick]] = True
        neg = neg.reshape(neg_pool.shape)
    return dict(pos=pos, neg=neg, n_pos=n_pos, n_neg=int(neg.sum()), n_neg_pool=n_pool,
                neg_ratio_requested=float(neg_ratio), negatives_capped=bool(want >= n_pool))


# --------------------------------------------------------------------------------- selection
def shaped(prob: np.ndarray, t0: float, thin: bool, dilate: int, R: int) -> np.ndarray:
    q = floor_sharpen(prob, t0=t0, hard=True)
    if thin:
        q = dominant_thin(q, R=R, p=prob)
    if dilate:
        q = dilate_mask(q, radius=int(dilate))
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def candidate_grid(floors: int = FLOOR_GRID, dilates=DILATE_GRID):
    grid = list(np.geomspace(1e-4, 0.9, floors))
    out = []
    for t0 in [0.0] + [float(f"{x:.6g}") for x in grid]:
        for thin in (False, True):
            for d in dilates:
                out.append((float(t0), bool(thin), int(d)))
    return out


def eligibility(row: dict, footprint_px: int, max_fraction: float = MAX_EMITTED_FRACTION,
                min_px: int = MIN_EMITTED_PX) -> tuple[bool, str]:
    frac = row["emitted_px"] / max(1, footprint_px)
    if row["emitted_px"] > max_fraction * footprint_px:
        return False, (f"emitted {row['emitted_px']:,} px = {frac:.1%} of the footprint "
                       f"> cap {max_fraction:.0%}")
    if row["emitted_px"] < min_px:
        return False, f"emitted {row['emitted_px']:,} px < floor {min_px:,}"
    return True, "within the pre-registered support window"


# --------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/training_features.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--out-dir", default="data/evidence/newfault")
    ap.add_argument("--fold", type=int, default=0, help="fold that SELECTS the emission policy")
    ap.add_argument("--eval-fold", type=int, default=1, help="fold that MEASURES it")
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-mode", default="balanced", choices=("balanced", "contiguous"))
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--min-samples-leaf", type=int, default=40)
    ap.add_argument("--l2-regularization", type=float, default=1.0)
    ap.add_argument("--neg-ratio", type=float, default=10.0)
    ap.add_argument("--max-negatives", type=int, default=1_200_000)
    ap.add_argument("--supervision", default="union",
                    choices=("union", "proxy_only", "catalogue"),
                    help="positive-label population: 'union' = catalogue∪proxy (default); "
                         "'proxy_only' = SGMC code-2 only (structurally anti-correlated with "
                         "catalogue-trained members); 'catalogue' = labels.tif only")
    ap.add_argument("--corridor", type=int, default=0, metavar="PX",
                    help="widen the positive set to the metric's own tolerance region: every pixel "
                         "within PX of a labelled fault becomes a positive. 0 (default) trains on "
                         "the traces themselves; 3 trains the classifier on exactly the quantity "
                         "the scorer rewards, using src.metrics.kernel_offsets as the structuring "
                         "element so the disk is the scorer's, not an approximation of it")
    ap.add_argument("--use-proxy-labels", action="store_true", default=None,
                    help="deprecated synonym for --supervision union (kept for reproduce cmds)")
    ap.add_argument("--no-proxy-labels", dest="use_proxy_labels", action="store_false",
                    default=None,
                    help="deprecated synonym for --supervision catalogue")
    return ap


def resolve_supervision(args) -> str:
    """Map the new --supervision flag and the legacy --[no-]proxy-labels flags to one mode.

    Precedence: an explicit legacy flag wins only when --supervision was left at its
    default AND the legacy flag was actually passed.  Mixing an explicit non-default
    --supervision with a contradictory legacy flag is a hard error, not a silent pick.
    """
    legacy = getattr(args, "use_proxy_labels", None)
    mode = str(getattr(args, "supervision", "union") or "union")
    if legacy is None:
        return mode
    legacy_mode = "union" if legacy else "catalogue"
    # If the caller only used the legacy flag, honour it.  If they also set --supervision
    # to something other than the default, the two must agree.
    if mode == "union" and legacy_mode == "catalogue":
        return "catalogue"          # pure --no-proxy-labels path
    if mode == "union" and legacy_mode == "union":
        return "union"              # pure --use-proxy-labels (or default) path
    if mode != legacy_mode and not (mode == "proxy_only"):
        raise SystemExit(
            f"conflicting supervision flags: --supervision={mode} vs "
            f"--{'use' if legacy else 'no'}-proxy-labels; pick one")
    return mode


def build_positives(fault: np.ndarray, proxy: np.ndarray, proxy_near: np.ndarray,
                    mode: str, corridor: int = 0) -> np.ndarray:
    """Positive mask for the requested supervision mode.

    * union       — catalogue ∪ proxy-near ∪ proxy-only (production default)
    * proxy_only  — proxy code-2 ONLY; never a catalogue fault.  This is the
                    structurally different member the LOO finding asked for.
    * catalogue   — catalogue only (ablation / legacy --no-proxy-labels)

    ``corridor`` > 0 widens the positive set to the metric's OWN tolerance region: the ground
    truth pixel g earns credit from any prediction within R = 3 px (k(d) = (1-d/R)_+), so the
    quantity a member is asked to reproduce is not "this pixel is a fault" but "a fault is within
    300 m of this pixel".  Training on the dilated positives asks the classifier that question
    directly.  The structuring element is built from ``src.metrics.kernel_offsets`` so the
    corridor is the SAME Euclidean disk the scorer uses, not a 4- or 8-connected approximation
    of it; a diamond would include (3, 0) and miss (2, 2), which the metric does not.
    """
    if mode == "union":
        base = fault | proxy | proxy_near
    elif mode == "proxy_only":
        base = proxy.copy()
    elif mode == "catalogue":
        base = fault.copy()
    else:
        raise ValueError(f"unknown supervision mode: {mode!r}")
    if not corridor:
        return base
    R = int(corridor)
    foot = np.zeros((2 * R + 1, 2 * R + 1), bool)
    for dy, dx, _k in kernel_offsets(R):
        foot[dy + R, dx + R] = True
    return ndimage.binary_dilation(base, structure=foot, iterations=1)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    supervision = resolve_supervision(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    fault, valid = read_truth(Path(args.labels))
    footprint, profile = read_template(Path(args.template))
    with rasterio.open(args.template) as _t:   # the raster itself: the mask the platform enforces
        template = _t.read(1).astype(np.float32)
    proxy = read_code_mask(Path(args.proxy), 2) & footprint
    proxy_near = read_code_mask(Path(args.proxy), 1) & footprint
    H, W = fault.shape
    shape_hw = (H, W)

    ids = block_id_map(shape_hw, args.block_px)
    table = block_table(shape_hw, args.block_px, valid=footprint, labels=fault)
    fold_of = assign_folds(table, args.folds, args.seed, args.block_mode)
    partition = describe_partition(shape_hw, args.block_px, args.folds, args.seed, table,
                                   fold_of, buffer_px=DEFAULT_R_PIXELS, labels=fault,
                                   valid=footprint, mode=args.block_mode)

    # --- training geography: everything except the two held-out folds (plus their R collars)
    excluded = (held_out_mask(shape_hw, args.block_px, fold_of, args.fold,
                              buffer_px=DEFAULT_R_PIXELS)
                | held_out_mask(shape_hw, args.block_px, fold_of, args.eval_fold,
                                buffer_px=DEFAULT_R_PIXELS))
    trainable = valid & ~excluded

    positives = build_positives(fault, proxy, proxy_near, supervision,
                               corridor=int(getattr(args, "corridor", 0) or 0))
    # proxy_only mode still needs a non-empty positive set inside the trainable region;
    # fail loudly rather than fitting a classifier on zero positives.
    n_pos_trainable = int((positives & trainable).sum())
    if n_pos_trainable < 100:
        raise SystemExit(
            f"supervision={supervision!r} leaves only {n_pos_trainable} trainable positives "
            f"(need ≥100); check --proxy / --fold / --eval-fold")
    sel = choose_samples(positives, trainable, rng, args.neg_ratio, args.max_negatives)
    wanted = sel["pos"] | sel["neg"]
    y = wanted & positives

    X, idx = build_feature_rows(Path(args.features), wanted)
    yv = y.ravel()[idx].astype(np.int8)
    keep = ~np.isnan(X).all(axis=1)
    X, yv = X[keep], yv[keep]
    print(f"[nff] sampled {X.shape[0]:,} px ({int(yv.sum()):,} positive) x {X.shape[1]} features "
          f"({int((~keep).sum()):,} all-NaN rows dropped)", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=int(args.max_iter), learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes), min_samples_leaf=int(args.min_samples_leaf),
        l2_regularization=float(args.l2_regularization), early_stopping=False,
        random_state=int(args.seed))
    t0 = time.time()
    clf.fit(X, yv)
    fit_s = round(time.time() - t0, 1)
    print(f"[nff] fitted in {fit_s}s", flush=True)

    prob, pred_s = predict_grid(clf, Path(args.features), footprint, valid)
    common = dict(height=H, width=W, crs=profile.get("crs"),
                  transform=profile.get("transform"), nodata=float("nan"))
    prob_path = out_dir / "prob_raw.tif"
    write_submission(prob_path, prob, clean_profile(profile, **common),
                     band_description="new-fault probability")

    # --- contexts (global geometry, so a fold restriction cannot move a pixel's credit)
    ctx = {"catalogue": GtContext(fault, DEFAULT_R_PIXELS),
           "proxy": GtContext(proxy, DEFAULT_R_PIXELS)}
    footprint_px = int(footprint.sum())

    def sweep(scope_mask, scope_name):
        """Score every candidate policy ON `scope_mask` — the folds the sweep is allowed to see.

        THE BUG THIS FIXES (2026-09-26, session 34).  `scope_mask` was passed in and never used:
        every row was scored with `c.score(q)` over the WHOLE grid, while the rows were labelled
        `f"fold {args.fold} (selection)"` and the report said the scope was that fold.  A whole-grid
        score is not a superset that is merely noisier - it is a different measurement, because the
        members are trained on everything except folds 0 and 1, so folds 2 and 3 carry truth this
        detector has already seen.  Selecting on it is selection on in-sample geography wearing an
        out-of-sample label.  `scripts/newfault_detector.py`'s committed runs (seed42-45, po46) were
        produced by the old path; their reports say so, and re-running them is queued in
        SUGGESTIONS.md rather than silently re-made here.
        """
        rows = []
        for t0, thin, dil in candidate_grid():
            q = shaped(prob, t0, thin, dil, DEFAULT_R_PIXELS)
            row = dict(t0=t0, thin=thin, dilate=dil,
                       emitted_px=int((q > 0).sum()))
            for pop, c in ctx.items():
                row[f"{pop}_dti"] = score_within_mask(q, c, scope_mask)["dti"]
                row[f"{pop}_dti_whole"] = c.score(q)
            row["scope"] = scope_name
            ok, why = eligibility(row, footprint_px)
            row["eligible"] = ok
            row["eligibility_reason"] = why
            rows.append(row)
        return rows

    select_rows = sweep(scored_mask(shape_hw, args.block_px, fold_of, args.fold),
                        f"fold {args.fold} (selection)")
    elig = [r for r in select_rows if r["eligible"]]
    winner = max(elig, key=lambda r: (r["proxy_dti"], -r["dilate"], -r["emitted_px"]))
    print(f"[nff] selected policy t0={winner['t0']} thin={winner['thin']} "
          f"dilate={winner['dilate']} proxy_dti={winner['proxy_dti']:.4f}", flush=True)

    field = shaped(prob, winner["t0"], winner["thin"], winner["dilate"], DEFAULT_R_PIXELS)
    # floor_sharpen turns "NaN >= t0" into False, so the shaping destroys the survey footprint and
    # leaves 0.0 (finite) where the template demands NaN - the exact placement the platform rejects
    # with "Predicted values must be in range [0, 1]".  Conform before writing, and record what it
    # changed, exactly as scripts/sanitize_submission.py does for the deep-ensemble artifact.
    conformed, conform_stats = conform_to_template(field, template)
    sub_path = out_dir / "submission.tif"
    write_submission(sub_path, conformed, clean_profile(profile, **common),
                     band_description="fault probability")
    (out_dir / "sanitize.json").write_text(json.dumps(dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/newfault_detector.py (via src/submission_io.conform_to_template)",
        why=("the shaping step maps NaN -> 0.0 outside the survey footprint; conform_to_template "
             "restores the template's mask (finite inside, NaN outside, nodata=nan)"),
        template=dict(path=args.template, sha256=sha256_file(Path(args.template))),
        after=dict(path=str(sub_path), sha256=sha256_file(sub_path),
                   bytes=sub_path.stat().st_size,
                   findings=dict(conformant=True,
                                 nan_px=int(np.isnan(conformed).sum()),
                                 finite_px=int(np.isfinite(conformed).sum()))),
        changes=conform_stats,
        scoring_impact=("none by construction: src/metrics.py scores through "
                        "np.nan_to_num(..., nan=0.0) and every scored population is a subset of "
                        "the template's valid region"),
    ), indent=1) + "\n")

    # --- unbiased measurement on geography the sweep never scored
    gen = {}
    for f_name, mask in [("selection", scored_mask(shape_hw, args.block_px, fold_of, args.fold)),
                         ("measurement", scored_mask(shape_hw, args.block_px, fold_of,
                                                     args.eval_fold)),
                         ("whole grid", np.ones(shape_hw, bool))]:
        gen[f_name] = {pop: score_within_mask(conformed, c, mask) for pop, c in ctx.items()}

    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/newfault_detector.py",
        purpose=("new-fault-first detector: lineament features + catalogue/independent-compilation "
                 "supervision, with the emission policy selected on the NEW-FAULT population on "
                 "geography the model never saw"),
        inputs=dict(
            features=dict(path=args.features, sha256=sha256_file(Path(args.features))),
            labels=dict(path=args.labels, sha256=sha256_file(Path(args.labels))),
            proxy=dict(path=args.proxy, sha256=sha256_file(Path(args.proxy))),
            template=dict(path=args.template, sha256=sha256_file(Path(args.template)))),
        metric=dict(R_pixels=DEFAULT_R_PIXELS, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA),
        supervision=dict(mode=supervision,
                         # Legacy boolean kept so existing consumers of the report keep working.
                         # True for any mode that draws positives from the proxy compilation.
                         use_proxy_labels=(supervision in ("union", "proxy_only")),
                         positive_px=int(positives.sum()),
                         catalogue_px=int(fault.sum()),
                         proxy_only_px=int(proxy.sum()),
                         proxy_near_px=int(proxy_near.sum()),
                         catalogue_in_positives=bool(supervision in ("union", "catalogue")),
                         proxy_only_in_positives=bool(supervision in ("union", "proxy_only")),
                         # Session 34: the corridor is part of the TARGET, so a member's row is only
                         # interpretable if the width is recorded next to the mode.  `corridor_px`
                         # is the raw trace count before dilation, so a reader can see how much the
                         # positive set grew (a 3 px disk multiplies a 1 px trace network ~7x).
                         corridor_px=int(getattr(args, "corridor", 0) or 0),
                         positives_before_corridor_px=(int(build_positives(
                             fault, proxy, proxy_near, supervision).sum())
                             if getattr(args, "corridor", 0) else int(positives.sum())),
                         note=({"union": "catalogue ∪ proxy-near ∪ proxy-only (default)",
                                "proxy_only": ("SGMC code-2 ONLY — structurally anti-correlated "
                                               "with catalogue-trained detectors"),
                                "catalogue": "labels.tif only (ablation)"}[supervision])),
        partition=partition,
        held_out_fold=args.fold,
        measurement_fold=args.eval_fold,
        training=dict(model="HistGradientBoostingClassifier", seed=args.seed,
                      max_iter=args.max_iter, learning_rate=args.learning_rate,
                      max_leaf_nodes=args.max_leaf_nodes,
                      min_samples_leaf=args.min_samples_leaf,
                      l2_regularization=args.l2_regularization,
                      n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
                      n_pos=int(sel["n_pos"]), n_neg=int(sel["n_neg"]),
                      n_neg_pool=int(sel["n_neg_pool"]),
                      neg_ratio_requested=sel["neg_ratio_requested"],
                      negatives_capped=sel["negatives_capped"]),
        timings_s=dict(fit=fit_s, predict=pred_s),
        policy_selection=dict(
            population="proxy (new-fault-like) pixels only - the population rules SS1.1 scores",
            scope=f"blocks of fold {args.fold} (spatially held out of training)",
            selection_key="proxy_dti (scored ON the selection fold's blocks since 2026-09-26)",
            ranking_key=("proxy_dti desc on the SELECTION FOLD's blocks, then narrower band, then "
                         "fewer emitted pixels; `proxy_dti_whole`/`catalogue_dti_whole` are "
                         "reported for context and never used to pick"),
            winner=winner, candidates=select_rows),
        generalisation=dict(
            fold=args.eval_fold,
            scope=(f"blocks of fold {args.eval_fold}: excluded from training and never used "
                   "by the policy sweep"),
            by_scope=gen),
        submission=dict(
            path=str(sub_path), bytes=sub_path.stat().st_size,
            sha256=sha256_file(sub_path), width=W, height=H, count=1, dtype="float32",
            crs=str(profile.get("crs")), res=[100.0, 100.0],
            transform=list(profile.get("transform"))[:6],
            finite_px=int(np.isfinite(conformed).sum()), total_px=int(conformed.size),
            nan_px=int(np.isnan(conformed).sum()),
            nonzero_px=int((np.nan_to_num(conformed, nan=0.0) > 0).sum()),
            policy=f"t0{winner['t0']}_thin{int(winner['thin'])}_w{winner['dilate']}",
            sanitize_evidence=str(out_dir / "sanitize.json"),
            global_scores={pop: ctx[pop].score(conformed) for pop in ctx}),
        prob_raw=dict(path=str(prob_path), bytes=prob_path.stat().st_size,
                      sha256=sha256_file(prob_path),
                      finite_px=int(np.isfinite(prob).sum())),
        reproduce=(f"python scripts/newfault_detector.py --features {args.features} "
                   f"--labels {args.labels} --proxy {args.proxy} --template {args.template} "
                   f"--out-dir {out_dir} --fold {args.fold} --eval-fold {args.eval_fold} "
                   f"--seed {args.seed} --max-iter {args.max_iter} "
                   f"--neg-ratio {args.neg_ratio} "
                   f"--supervision {supervision}"),
        caveats=[
            "Every number is a local surrogate on held-out geography; the real metric is the "
            "private expert-labelled new-fault set (rules SS1.1 / SS3.2).",
            "The proxy population is an independent public compilation (SGMC), not the scored "
            "faults; it is the closest available stand-in, and selecting on it is itself a "
            "modelling assumption that the next session should re-test against the public "
            "leaderboard.",
            "The catalogue DTI is reported for context only. Rules SS1.1/SS3.2 score the new "
            "faults, so a policy that is worse on the catalogue is not thereby disqualified.",
            ("proxy_only supervision never sees a catalogue fault as a positive: its catalogue "
             "DTI is expected to be lower than a union-supervised member, and that is the point "
             "- diversity of *errors*, not diversity of seeds."
             if supervision == "proxy_only" else
             "Default supervision is catalogue ∪ proxy; --supervision proxy_only is the "
             "structurally different member for the union."),
        ],
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[nff] wrote {out_dir/'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
