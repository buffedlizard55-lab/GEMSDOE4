#!/usr/bin/env bash
# Session 34 (2026-09-26) — re-select the union's combination rule on the SELECTION FOLD, and audit
# whether the union's folds are a common holdout for its members.
#
# Reproduces, in order:
#   1. the fold-discipline audit          -> data/evidence/fold_discipline.json
#   2. the production union re-selection  -> data/evidence/combined/{submission.tif,report.json}
#      (selection key = the SELECTION fold's proxy DTI; the whole grid is context, never a key)
#   3. the clean-pool CONTROL run         -> data/evidence/combined_clean/  (every member held out
#      both union folds; the arm that can be measured without qualification)
#   4. the metric's emission budget       -> data/evidence/emission_budget.json
#   5. provenance: validate + package + payload + readiness, and rebuild the site
#
# Every step is safe to re-run: the combiner is deterministic given the committed member rasters,
# so re-running step 2 must reproduce the same submission.tif bytes (sha 237f0063…).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

FIVE=(--member deep11=data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif
      --member classical=data/evidence/baseline/submission.tif
      --member nff43=data/evidence/newfault/seed43/prob_raw.tif:prob
      --member nff45=data/evidence/newfault/seed45/prob_raw.tif:prob
      --member po46=data/evidence/newfault/proxy_only46/prob_raw.tif:prob)

echo "== 1. fold discipline =="
python scripts/audit_fold_discipline.py

echo "== 2. production union, selected on the selection fold =="
python scripts/combine_newfault.py \
  --labels data/labels.tif --proxy data/evidence/proxy/proxy_catalogue.tif \
  --template data/sample_submission.tif \
  --fold 0 --eval-fold 1 --floors 24 --dilates 0,1 --votes 1,2,3,4,5 \
  --out-dir data/evidence/combined --name "nff-k3-fold0-selected" "${FIVE[@]}"
python scripts/validate_submission.py --pred data/evidence/combined/submission.tif

echo "== 3. clean-pool control (members that held out BOTH union folds) =="
python scripts/combine_newfault.py \
  --labels data/labels.tif --proxy data/evidence/proxy/proxy_catalogue.tif \
  --template data/sample_submission.tif \
  --fold 0 --eval-fold 1 --floors 24 --dilates 0,1 --votes 1,2,3,4,5 \
  --out-dir data/evidence/combined_clean --name "cleanpool-control" \
  --member classical=data/evidence/baseline/submission.tif \
  --member nff42=data/evidence/newfault/seed42/prob_raw.tif:prob \
  --member nff44=data/evidence/newfault/seed44/prob_raw.tif:prob \
  --member po46=data/evidence/newfault/proxy_only46/prob_raw.tif:prob

echo "== 4. the metric's own emission budget on the shipped bytes =="
python scripts/emission_budget.py --pred data/evidence/combined/submission.tif \
  --out data/evidence/emission_budget.json

echo "== 5. provenance =="
python scripts/build_submission_payload.py
python scripts/check_site_generator.py
python scripts/package_submission.py
python scripts/check_submission_readiness.py
python scripts/build_site.py
echo "DONE"
