#!/usr/bin/env bash
# Session 32 — proxy_only diversity member + LOO re-adoption of the union.
# Pre-registered BEFORE any new measurement (see STATUS.md Session 32):
#   * train one NFF member with --supervision proxy_only (seed 46, folds 0/1)
#   * recover the classical baseline member (sha pin 9f2577cf…)
#   * re-run combine_newfault over the previous 5-member set PLUS proxy_only46
#   * LOO drop each member; adopt if measurement-fold proxy DTI beats the
#     committed nff-union-5 (0.1897) by ≥ +0.010 OR paired P ≥ 0.95
#   * otherwise KEEP the committed artifact and record the negative result
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

COMBINE=(python scripts/combine_newfault.py
  --labels data/labels.tif
  --proxy data/evidence/proxy/proxy_catalogue.tif
  --template data/sample_submission.tif
  --fold 0 --eval-fold 1 --floors 18 --votes 1,2 --dilates 0
  --member deep11=data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif
  --member classical=data/evidence/baseline/submission.tif
  --member nff42=data/evidence/newfault/seed42/prob_raw.tif:prob
  --member nff43=data/evidence/newfault/seed43/prob_raw.tif:prob
  --member nff45=data/evidence/newfault/seed45/prob_raw.tif:prob
  --member po46=data/evidence/newfault/proxy_only46/prob_raw.tif:prob
)

echo "== full 6-member (5 prior + proxy_only) =="
"${COMBINE[@]}" --out-dir data/evidence/union_po --name nff-po-union-6

for drop in deep11 classical nff42 nff43 nff45 po46; do
  echo "== LOO drop $drop =="
  args=()
  for m in deep11 classical nff42 nff43 nff45 po46; do
    [[ "$m" == "$drop" ]] && continue
    case "$m" in
      deep11)    args+=(--member deep11=data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif) ;;
      classical) args+=(--member classical=data/evidence/baseline/submission.tif) ;;
      nff42)     args+=(--member nff42=data/evidence/newfault/seed42/prob_raw.tif:prob) ;;
      nff43)     args+=(--member nff43=data/evidence/newfault/seed43/prob_raw.tif:prob) ;;
      nff45)     args+=(--member nff45=data/evidence/newfault/seed45/prob_raw.tif:prob) ;;
      po46)      args+=(--member po46=data/evidence/newfault/proxy_only46/prob_raw.tif:prob) ;;
    esac
  done
  python scripts/combine_newfault.py \
    --labels data/labels.tif \
    --proxy data/evidence/proxy/proxy_catalogue.tif \
    --template data/sample_submission.tif \
    --fold 0 --eval-fold 1 --floors 18 --votes 1,2 --dilates 0 \
    --out-dir "data/evidence/union_po_loo/drop_${drop}" \
    --name "nff-po-drop-${drop}" \
    "${args[@]}"
done
echo "DONE session32 LOO"
