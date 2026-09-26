#!/usr/bin/env bash
# Session 34 (2026-09-26) — the fold-scoped selection fix, the vote axis, and the k = 3 adoption.
#
# Reproduces the whole decision from committed inputs, in the order it was made:
#   1. snapshot the artifact currently shipped, as the REFERENCE every contrast is paired against;
#   2. re-run the union with its own (pre-registered) defaults: --floors 24 --dilates 0,1
#      --votes 1,2,3,4,5, selecting on the SELECTION fold's blocks only (SELECTION_KEY);
#   3. validate the new bytes against the official template (the [0,1]/NaN gate);
#   4. contrast new vs old, paired, on fold 1 (never swept) and on the pooled honest pool (folds 0+1);
#   5. print the adoption verdict against the pre-registered test in docs/SESSION34_PROTOCOL.md.
#
# Nothing here is a leaderboard prediction: the truth is the SGMC proxy compilation, and the report
# says so in every file it writes.  See data/evidence/combined/report.json for the full 240-row
# sweep, and data/evidence/emission_budget.json for what a higher target would cost.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

SHIPPED=data/evidence/combined/submission.tif
PREV=data/evidence/union_po_loo/prev_committed_$(date -u +%Y%m%dT%H%M%SZ).tif

echo "== 1. snapshot the current artifact as the paired reference =="
sha256sum "$SHIPPED"
cp "$SHIPPED" "$PREV"
echo "   reference -> $PREV"

echo "== 2. re-run the union under the fixed selection rule =="
python scripts/combine_newfault.py \
  --labels data/labels.tif \
  --proxy data/evidence/proxy/proxy_catalogue.tif \
  --template data/sample_submission.tif \
  --fold 0 --eval-fold 1 \
  --member deep11=data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif \
  --member classical=data/evidence/baseline/submission.tif \
  --member nff43=data/evidence/newfault/seed43/prob_raw.tif:prob \
  --member nff45=data/evidence/newfault/seed45/prob_raw.tif:prob \
  --member po46=data/evidence/newfault/proxy_only46/prob_raw.tif:prob \
  --out-dir data/evidence/combined \
  --name "nff-k3-fold0-selected"

echo "== 3. validate the new bytes =="
python scripts/validate_submission.py --pred "$SHIPPED"

echo "== 4a. paired contrast on fold 1 (the fold the sweep never scored) =="
python scripts/paired_union_contrast.py \
  --reference "$PREV" --candidate "$SHIPPED" --population proxy \
  --fold 1 --note "fold 1 only: never used by the sweep, and out-of-sample for every NFF member" \
  --out data/evidence/union_po_loo/contrasts/k3_vs_prev_fold1.json

echo "== 4b. paired contrast pooled over the honest pool (folds 0+1) =="
python scripts/paired_union_contrast.py \
  --reference "$PREV" --candidate "$SHIPPED" --population proxy \
  --fold 0,1 \
  --note "POOLED honest pool: folds 0+1 are the two folds scripts/newfault_detector.py excluded from every NFF member's training. DISCLOSURE: fold 0 is the fold the sweep selected on, so its share of the gain is selection-biased upward; fold 1 alone is the clean read. Folds 2/3 are in-sample for the members and are NOT pooled here." \
  --out data/evidence/union_po_loo/contrasts/k3_vs_prev_pooled01.json

echo "== 5. the metric's own budget for the new bytes =="
python scripts/emission_budget.py --pred "$SHIPPED" --out data/evidence/emission_budget.json

echo "DONE. Adopt only if the fold-1 and pooled contrasts agree in sign and the interval on the"
echo "honest pool excludes zero (docs/SESSION34_PROTOCOL.md §4); otherwise restore the reference:"
echo "  cp $PREV $SHIPPED"
