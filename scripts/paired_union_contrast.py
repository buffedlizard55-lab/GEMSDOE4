#!/usr/bin/env python3
"""Paired block-bootstrap contrast of two submission rasters on one fold's blocks.

WHY THIS EXISTS
---------------
Adopting a new union artifact is a decision, and the repository's rule for adoption
(docs/FIELD_SELECTION_RULE.md R3, applied here to the union line) wants a PAIRED
probability, not two point estimates shouted next to each other.  Both rasters are
scored on the SAME blocks of the SAME fold with the SAME truth, the DTI is recomposed
from the additive per-block components under block resampling
(`src.metrics.bootstrap_from_blocks`), and the output is
`prob_beats_reference` - the bootstrap probability that the candidate beats the
reference on the same resample.

The truth is the SGMC proxy compilation (code 2), the same stand-in for the private
expert-mapped new faults every new-fault number in this repository uses; the script
records that fact in its output so the JSON cannot be quoted out of context.

IDENTITY CHECK (the guard against scoring the wrong thing)
----------------------------------------------------------
`block_aggregate` groups the GLOBAL metric components by block, so summing the kept
blocks' TP_w/FP_w/FN_w must reproduce `score_within_mask` on the same scope exactly.
The script asserts that identity and refuses to emit a bootstrap if it fails, so the
JSON cannot describe a different measurement than the scalar it claims to contrast.

POOLING SEVERAL FOLDS (`--fold 2,3`)
------------------------------------
One fold of the 512 px partition yields ~9 resampling units, which is below this
repository's readable-CI threshold (`MIN_BLOCKS_FOR_A_READABLE_CI` = 12): a bootstrap
that redraws 9 units cannot separate a regional preference from block-level noise.
`--fold` therefore accepts a comma-separated list, and the bootstrap then redraws the
POOLED blocks of all the named folds, while `per_fold` still reports each fold's own
scalar pair.  Pooling is only legitimate because the resampling unit is the block and
the blocks of different folds are disjoint: the pooled interval is a statement about
"the union of these regions", not about any one of them.  Record which folds were
selection folds with `--note` - a fold the sweep selected on is biased upward and must
never be pooled silently with folds nothing touched.

    python scripts/paired_union_contrast.py \
        --reference data/evidence/combined/submission.tif \
        --candidate data/evidence/union6_loo/drop_nff44/submission.tif \
        --fold 1 --out data/evidence/union6/paired_contrast.json

    # pooled over the two folds the selection sweep never saw (18 blocks > the 12-unit bar)
    python scripts/paired_union_contrast.py \
        --reference ... --candidate ... --fold 2,3 \
        --note "folds 2 and 3 were used by neither the selection nor the measurement sweep" \
        --out data/evidence/union_po_loo/contrasts/pooled_folds23.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import (assign_folds, block_id_map, block_table,  # noqa: E402
                        scored_mask)
from src.metrics import (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_R_PIXELS,  # noqa: E402
                         EPS, GtContext, block_aggregate, bootstrap_from_blocks,
                         score_within_mask)


def read_field(path: Path) -> np.ndarray:
    """Submission raster as the scorer sees it: NaN/nodata -> 0, clipped to [0, 1]."""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = 0.0
    return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--proxy", default="data/evidence/proxy/proxy_catalogue.tif")
    ap.add_argument("--template", default="data/sample_submission.tif")
    ap.add_argument("--population", default="proxy", choices=("proxy", "catalogue"),
                    help="which truth the contrast is measured against (default: the "
                         "proxy compilation that stands in for the private new faults)")
    ap.add_argument("--fold", default="1",
                    help="the fold whose blocks are scored, or a comma-separated list to pool "
                         "them (e.g. '2,3'); pooling raises the resampling-unit count above the "
                         "12-block readable-CI threshold")
    ap.add_argument("--note", default="",
                    help="provenance recorded verbatim in the JSON - say here which of the "
                         "pooled folds the selection sweep saw, if any")
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-mode", default="balanced", choices=("balanced", "contiguous"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap


def parse_fold_list(spec: str, n_folds: int) -> list:
    """`'1'` -> [1]; `'2,3'` -> [2, 3]; anything else (dupes, out of range, junk) is an error.

    A silently-dropped fold would pool fewer blocks than the caller asked for, which is exactly
    the kind of quiet undercount this repository's readable-CI rule exists to prevent.
    """
    tokens = str(spec).replace(" ", "").split(",")
    if any(t == "" for t in tokens):
        raise SystemExit(f"--fold has an empty entry: {spec!r} (a trailing or doubled comma is a "
                         "typo, not a fold - silently dropping it would pool fewer blocks "
                         "than asked for)")
    try:
        folds = [int(t) for t in tokens]
    except ValueError:
        raise SystemExit(f"--fold must be an integer or a comma-separated list, got {spec!r}")
    if not folds:
        raise SystemExit("--fold must name at least one fold")
    if len(set(folds)) != len(folds):
        raise SystemExit(f"--fold names a fold twice: {spec!r} (a duplicated fold would double "
                         "count its blocks in the bootstrap)")
    bad = [f for f in folds if not 0 <= f < n_folds]
    if bad:
        raise SystemExit(f"--fold names folds outside the partition 0..{n_folds - 1}: {bad}")
    return sorted(folds)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    fold_list = parse_fold_list(args.fold, int(args.folds))

    with rasterio.open(args.labels) as src:
        lab = src.read(1)
        nodata = src.nodata
    valid = np.ones(lab.shape, bool) if nodata is None else (lab != nodata)
    fault = (lab == 1) & valid
    with rasterio.open(args.template) as src:
        template = src.read(1)
    footprint = np.isfinite(template)
    with rasterio.open(args.proxy) as src:
        proxy_code = src.read(1)
    proxy = (proxy_code == 2) & footprint

    ctx = {"proxy": GtContext(proxy, DEFAULT_R_PIXELS),
           "catalogue": GtContext(fault, DEFAULT_R_PIXELS)}
    truth = {"proxy": proxy, "catalogue": fault}[args.population]

    H, W = fault.shape
    table = block_table((H, W), args.block_px, valid=footprint, labels=fault)
    fold_of = assign_folds(table, args.folds, args.seed, args.block_mode)
    blocks = block_id_map((H, W), args.block_px)
    # scope = the union of the named folds' blocks (disjoint by construction of the partition);
    # block_fold says which named fold each block belongs to, so the pooled table can still be
    # broken out per fold and a reader can see which region each resampling unit came from
    scope = np.zeros((H, W), bool)
    fold_scopes = {}
    for f in fold_list:
        m = scored_mask((H, W), args.block_px, fold_of, f)
        fold_scopes[f] = m
        scope |= m
    block_fold = {}
    for f in fold_list:
        for b in np.unique(blocks[fold_scopes[f]]):
            block_fold[int(b)] = f

    ref_sha = hashlib.sha256(Path(args.reference).read_bytes()).hexdigest()
    cand_sha = hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest()
    if ref_sha == cand_sha:
        raise SystemExit("reference and candidate are the same bytes - a contrast of a file "
                         "with itself is not a measurement (this guard exists because that "
                         "exact mistake happened once: after an adoption, the default "
                         "--reference path already pointed at the new artifact)")
    ref = read_field(Path(args.reference))
    cand = read_field(Path(args.candidate))

    agg_r = block_aggregate(ctx[args.population], ref, blocks, int(blocks.max()) + 1)
    agg_c = block_aggregate(ctx[args.population], cand, blocks, int(blocks.max()) + 1)
    del agg_r["credit"], agg_c["credit"]          # per-pixel vectors, not per-block rows

    # every block of this fold takes part: a block with no truth still carries the FP mass of
    # its own prediction pixels, and `score_within_mask` counts exactly that - dropping such
    # blocks would break the recomposition identity below (measured, not assumed: it failed
    # once already and the guard refused to emit a bootstrap)
    fold_ids = sorted({int(b) for b in np.unique(blocks[scope])})
    rows = []
    for b in fold_ids:
        rows.append({"block": b, "fold": block_fold.get(b),
                     "scoreable": int(agg_r["n_gt"][b]) > 0,
                     "TP_w": float(agg_r["TP"][b]), "FP_w": float(agg_r["FP"][b]),
                     "FN_w": float(agg_r["FN"][b]), "n_gt": int(agg_r["n_gt"][b])})
    # `scoreable` and `n_gt` are properties of the TRUTH, not of the arm: both rasters are scored
    # against the same pixels, so the candidate's rows must carry the same flags.  They did not
    # until 2026-09-26, and the committed adoption contrast therefore reported
    # `n_scoreable_blocks: 0` for the candidate arm next to its own non-zero DTI - a reported
    # number that read as "no truth here" on the arm that won.  Fixed, and pinned by
    # tests/test_paired_contrast.py.
    cand_rows = [{"scoreable": r["scoreable"],
                  "TP_w": float(agg_c["TP"][r["block"]]), "FP_w": float(agg_c["FP"][r["block"]]),
                  "FN_w": float(agg_c["FN"][r["block"]]), "n_gt": r["n_gt"]} for r in rows]

    # identity check: recomposing the fold's blocks must reproduce `score_within_mask` on the
    # scope - the same function the combiner's measurement column is produced by
    def recompose(per_block):
        tp = sum(r["TP_w"] for r in per_block)
        fp = sum(r["FP_w"] for r in per_block)
        fn = sum(r["FN_w"] for r in per_block)
        return tp / (tp + DEFAULT_ALPHA * fp + DEFAULT_BETA * fn + EPS)

    ref_scalar = score_within_mask(ref, ctx[args.population], scope)["dti"]
    cand_scalar = score_within_mask(cand, ctx[args.population], scope)["dti"]
    ident = dict(reference=abs(recompose(rows) - ref_scalar),
                 candidate=abs(recompose(cand_rows) - cand_scalar))
    TOL = 1e-9
    if max(ident.values()) > TOL:
        print(json.dumps(dict(identity=ident, ref_scalar=ref_scalar, cand_scalar=cand_scalar)))
        raise SystemExit("identity check FAILED - the per-block table does not recompose to "
                         "the scalar measurement; refusing to emit a bootstrap")

    # per-fold breakdown: each fold's own scalar pair, scored on that fold's blocks alone.  The
    # pooled bootstrap is the headline only because a pooled number hides which region carried
    # it, and a fold the selection sweep saw is biased upward - so both are reported.
    per_fold = []
    for f in fold_list:
        fr = score_within_mask(ref, ctx[args.population], fold_scopes[f])["dti"]
        fc = score_within_mask(cand, ctx[args.population], fold_scopes[f])["dti"]
        per_fold.append(dict(fold=f,
                             n_blocks=sum(1 for r in rows if r["fold"] == f),
                             reference_dti=round(float(fr), 6),
                             candidate_dti=round(float(fc), 6),
                             delta=round(float(fc - fr), 6)))

    boot = bootstrap_from_blocks(
        [dict(blocks=rows, label="reference"), dict(blocks=cand_rows, label="candidate")],
        n_boot=int(args.n_boot), seed=int(args.boot_seed))

    out = {
        "generated_by": "scripts/paired_union_contrast.py",
        "population": args.population,
        "population_note": ("SGMC proxy compilation (code 2), the stand-in for the privately "
                            "withheld expert-mapped new faults - not the scored set "
                            "(rules SS1.1/SS3.2)"),
        "fold": (fold_list[0] if len(fold_list) == 1 else None),
        "folds": fold_list,
        "fold_note": ("single fold" if len(fold_list) == 1 else
                      "POOLED over %d folds: the bootstrap redraws the disjoint blocks of all of "
                      "them, so the interval describes their union, not any one region"
                      % len(fold_list)),
        "per_fold": per_fold,
        "note": args.note or None,
        "partition": dict(block_px=int(args.block_px), folds=int(args.folds),
                          seed=int(args.seed), mode=args.block_mode),
        "reference": dict(path=args.reference, sha256=ref_sha,
                          scalar_dti=round(float(ref_scalar), 6)),
        "candidate": dict(path=args.candidate, sha256=cand_sha,
                          scalar_dti=round(float(cand_scalar), 6)),
        "identity_max_abs_error": {k: float(v) for k, v in ident.items()},
        "n_blocks_kept": len(rows),
        "n_boot": int(args.n_boot),
        "reliability": ("COARSE - %d resampling units is below this repository's readable-CI "
                        "threshold (12): the interval cannot separate a regional preference "
                        "from block-level noise, so treat P as suggestive and say so when "
                        "quoting it" % len(rows)) if len(rows) < 12
                       else "adequate - %d resampling units" % len(rows),
        "bootstrap": boot,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    b = boot.get("candidate", {})
    print(f"[contrast] folds {'+'.join(str(f) for f in fold_list)} "
          f"population={args.population} "
          f"blocks={len(rows)} ref={ref_scalar:.6f} cand={cand_scalar:.6f} "
          f"P(cand>ref)={b.get('prob_beats_reference')} "
          f"contrast_ci95={b.get('contrast_vs_reference_ci95')}")
    for pf in per_fold:
        print(f"[contrast]   fold {pf['fold']}: blocks={pf['n_blocks']} "
              f"ref={pf['reference_dti']:.6f} cand={pf['candidate_dti']:.6f} "
              f"delta={pf['delta']:+.6f}")
    print(f"[contrast] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
