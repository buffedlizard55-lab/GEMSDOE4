# Session 34 protocol — the selection scope, and the holdout the union never had

**Core values:** Maximize P(Win). Own the Outcome. Every number below has a command that produced it.

## The two defects this session found, and how

**Defect 1 — the sweep selected on a scope its own report said it did not use.**
`scripts/combine_newfault.py` ranked candidates by `proxy_dti` = `GtContext.score(field)`, the
**whole grid**, while its docstring, its `selection.scope` field and `docs/FIELD_SELECTION_RULE.md`
all said "the new-fault-population DTI on the SELECTION fold's blocks". Measured consequence: the
grid-scale statistic and the fold-scoped statistic disagree about which rule wins, and the disagreeing
cases are not near-ties. Fix: every candidate is now scored on four scopes
(`proxy_dti_selection`, `proxy_dti_measurement`, `proxy_dti_pooled01`, `proxy_dti_whole`) and exactly
one of them — `SELECTION_KEY = "proxy_dti_selection"` — selects. A scope with no truth pixels returns
`None` and `select_winner()` exits rather than ranking on a zero.

**Defect 2 — the union's folds were not a common holdout.**
Found by reading each member's own `report.json`, not by experiment: `nff43` and `nff45` were run with
`--fold 2 --eval-fold 3`, and `newfault_detector.py` trains on `valid & ~(held_out(fold) |
held_out(eval_fold))`. Those two members therefore **trained on folds 0 and 1** — the folds the union
selects and measures on. Three of the five shipped members are in-sample on the selection fold
(`nff43`, `nff45`, and the deep ensemble, which has no fold protocol at all).

## The measurements, in the order they were made

1. `scripts/audit_fold_discipline.py` — every committed NFF member scored at the SAME policy
   (`t0=0.5`, no dilation) on each fold, with a difference-in-differences estimator whose control
   group is the members whose own holdout *is* the union's folds. Measured (union-folds mean minus
   other-folds mean at that common policy): `seed43` **+0.0216**, `seed45` **+0.0060** — the two
   members trained on folds 0+1 — against controls `proxy_only46` **−0.1785**, `seed44` **−0.0860**,
   `seed42` **−0.0199**. Halved difference = **+0.0543 DTI** in-sample exposure, written to
   `data/evidence/fold_discipline.json`. The raw signature is starker than the estimate: at the same
   floor `proxy_only46` scores **0.187** on the folds it trained on and **0.009** on the folds it
   held out.
2. `scripts/combine_newfault.py` re-run on the pre-registered grid
   (`--floors 24 --dilates 0,1 --votes 1,2,3,4,5 --fold 0 --eval-fold 1`, 250 candidates). The
   fold-scoped argmax is `t0=0.124344, dilate=0, k=3 of 5`, selection 0.2856, measurement **0.2221**,
   pooled 0+1 0.2603 (the previous rule, k=2, measures 0.1990 on the same fold). Artifact
   `data/evidence/combined/submission.tif`, sha `237f0063a440…`, 505,882 B.
3. **Paired contrasts** against the previous artifact (`scripts/paired_union_contrast.py`, block
   bootstrap): fold 1 only (9 blocks) **0.199605 → 0.222076, Δ +0.022471, CI95 [−0.002626,
   +0.042000], P(better) 0.965**; pooled folds 0+1 (17 blocks) **0.247454 → 0.260266, Δ +0.012812,
   CI95 [−0.002834, +0.027379], P(better) 0.954**. Files:
   `data/evidence/union_po_loo/contrasts/k3_vs_k2_fold1.json`, `…_pooled01.json`; the reference
   raster is committed as `data/evidence/union_po_loo/prev_committed_nff_k2_t0.18.tif`
   (sha `c1da7dd9…`) so the contrast can be re-run from a fresh clone.
4. **The clean-pool control** (`data/evidence/combined_clean/`): the same search restricted to the
   four members that held out BOTH union folds (`classical`, `nff42`, `nff44`, `po46`). Honest
   numbers: selection-fold argmax is `k=4, t0=0.0172` (selection 0.1883) but that rule measures only
   0.0732 on the untouched fold — the best measured rules are `k=3` (0.1674) and `k=2` (0.1651).
5. `scripts/emission_budget.py` — the metric's own budget, inverted per scope
   (`data/evidence/emission_budget.json`). Measured on the shipped bytes: selection fold coverage
   47.7 % at FP/G 3.87 (DTI 0.2856), measurement fold 35.2 % at 3.58 (0.2221), pooled 0+1 42.6 % at
   3.75 (0.2603), whole grid 35.0 % at 3.55 (0.2213); the identity reproduces the scorer on every
   scope to ≤ 2.1e-16 relative error. On the pooled honest scope, reaching the current public leader
   number (0.3049) needs **+7.8 points of coverage** at the present false-positive price, or a **32 %
   cut in false-positive mass** at the present coverage - i.e. it is reachable on this surrogate, and
   it is arithmetic rather than a hope. On the clean-control pool the same target would need more
   coverage than the k = 3 field emits, which is what makes the pooled scope the informative one.

## Adoption decision, and the honest reading of it

The artifact was adopted because (a) it is the argmax of the pre-registered selection key, (b) both
paired contrasts point the same way, and (c) the k=3 family beats the k=2 family on the untouched
fold across the whole floor grid, not in one row. **The caveat is not hidden:** the pooled 95 %
interval includes zero by 0.0028, and the two members that make the 5-member pool strong are the two
that trained on the folds it is measured on. The clean-pool control shows the *level* of every number
above is optimistic (0.073–0.167 honest vs 0.222 with the contaminated pool) — which is why the
site quotes the clean-pool run next to the shipped one instead of only the better number.

**Not adopted, and why it is recorded anyway:** the clean pool's own selection-fold argmax (`k = 4`,
t0 = 0.0171793, selection 0.1883) measures **0.0732** on the untouched fold — 2.3x worse than the
family best there (`k = 3`: 0.1674, `k = 2`: 0.1651). A single-fold argmax over 200–250 candidates is
a noisy selector, and this is the measurement that shows it rather than an argument that says it.
Queued in `SUGGESTIONS.md`: select on a 1-SE band or on the pooled scope, pre-registered, and re-run.
Its bytes and hashes are committed (`data/evidence/combined_clean/`, sha `1b52d633…`, 474,736 B) —
a control arm that is described but not shipped is a claim, not evidence.

## What would falsify the shipped choice

1. A k=3 row that is *not* better than k=2 when both are measured on the clean pool — currently the
   clean pool gives 0.1674 (k=3) vs 0.1651 (k=2), a +0.0023 margin, i.e. **not** a decisive
   separation there. The k=2/k=3 distinction is supported by the contaminated pool (+0.023) and only
   weakly by the clean one.
2. Re-running `nff43` and `nff45` with `--fold 0 --eval-fold 1` and finding their union-equivalent
   pool no longer prefers k=3.
3. A third partition seed (all committed runs are seed 42; the falsifier has never been exercised).
