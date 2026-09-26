# Session 32 protocol — pre-registered before any new measurement (2026-09-26)

**Core values:** Maximize P(Win). Own the Outcome. No hallucinations — every number has a measurement.

## Why this session exists

Session 31 found that *more seeds of the same supervision* make the union worse
(`data/evidence/union6_loo/`): same features + same truth ⇒ correlated errors.
The next-session queue therefore asked for **member diversity from supervision,
not seeds**. QFaults was the first candidate third truth source; the committed
cross-catalogue transfer report already refused it
(`data/evidence/xcat/transfer_report.json`: 60,938 of 60,939 in-footprint QFaults
pixels lie within R = 3 px of a training label — B is the labels). That branch is
closed. The remaining free, public, independent signal is the SGMC proxy itself,
used as the *only* positive class.

## Pre-registered experiment

1. **Train one NFF member with `--supervision proxy_only`**
   - seed 46, folds 0 (select) / 1 (measure), max-iter 300, neg-ratio 10
   - positives = SGMC code-2 only (never a catalogue fault)
   - out: `data/evidence/newfault/proxy_only46/`
2. **Recover the classical baseline member** (sha pin `9f2577cf…`) so the
   5-member union is reproducible from a fresh clone.
3. **Re-combine** the previous adopted set (`deep11 ∪ classical ∪ nff42 ∪ nff43 ∪ nff45`)
   plus `po46`, with the same search the adoption used (`--floors 18 --votes 1,2`).
4. **Leave-one-out** drop each of the six members; record measurement-fold proxy DTI.
5. **Adoption rule (unchanged from session 31, written before any number):**
   adopt a candidate iff its measurement-fold proxy DTI beats the committed
   nff-union-5 (`0.1897`, sha `19de9950…`) by **margin ≥ +0.010 OR paired block
   bootstrap P(cand > ref) ≥ 0.95**. Otherwise KEEP the committed artifact and
   record the negative result as evidence.

## What is NOT allowed

- Tuning the floor grid, vote set, or folds after seeing the measurement number.
- Quoting selection-fold DTI as the headline.
- Claiming a leaderboard score (no DrivenData login in this sandbox).
- Treating the SGMC proxy as the scored set (rules §1.1 score expert-mapped new faults).

## Unique name / Note for any adopted submission

- File name pattern (browser builder): `gems-submission-<UTC>-<sha8>.tif`
- Suggested Note: `nff-po-union · proxy_only diversity · k=? of ? (t0=?, w=?)`
  (filled in from the combiner's `suggested_note` after measurement)

## Verified sources this protocol rests on

| claim | source |
|---|---|
| Phase 1 scores new faults | rules PDF §1.1, https://docs.nlr.gov/docs/fy26osti/96647.pdf |
| Phase 2 scores revised new faults | rules PDF §1.1 / §3.2 |
| QFaults ≈ labels (refused) | `data/evidence/xcat/transfer_report.json` |
| SGMC proxy is independent (61,664 px code-2) | `data/evidence/proxy/proxy_stats.json`, DOI 10.3133/ds1052 |
| Seed-only members dilute the union | `data/evidence/union6_loo/`, session 31 STATUS |
| Committed bar to beat | measurement fold 0.1897, sha `19de9950…` |


---

## Results (measured 2026-09-26, after the pre-registered run)

| config | selection fold proxy | measurement fold proxy | Δ vs 0.1897 | paired P | verdict |
|---|---:|---:|---:|---:|---|
| committed nff-union-5 | 0.2096 | **0.1897** | 0 | — | the bar |
| full 6 (+po46) | 0.2311 | 0.1905 | +0.0007 | — | REJECT |
| drop deep11 | 0.2325 | 0.1889 | −0.0008 | — | REJECT |
| drop classical | 0.2277 | 0.1886 | −0.0011 | — | REJECT |
| **drop nff42** | 0.2294 | **0.1996** | **+0.0099** | **1.0** | **ADOPT (P branch)** |
| drop nff43 | 0.2350 | 0.1893 | −0.0004 | — | REJECT |
| drop nff45 | 0.2339 | 0.1990 | +0.0093 | 0.7835 | REJECT |
| drop po46 (= identity) | 0.2096 | 0.1897 | 0 | — | recovers committed |

**Shipped:** `data/evidence/combined/submission.tif`
sha256 `c1da7dd9c44e05382b67367a8eba1987fca5594e33bf23812d308f37112dadb4` (547,082 B).
Members: `deep11 ∪ classical ∪ nff43 ∪ nff45 ∪ po46`, k = 2, t0 = 0.180482, w = 0.
Suggested Note: `nff-po-drop-nff42 · proxy_only diversity · union k=2 of 5 (t0=0.18, w=0)`.

**Honesty:** the interval is COARSE (9 blocks). The only unbiased test is a leaderboard upload.
