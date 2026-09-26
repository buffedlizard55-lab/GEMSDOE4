# GEMSDOE4 — Geologic Enhanced Mapping System (GEMS) Prize · new-fault-first submission line

> **THE FILE TO SUBMIT, IN ONE CLICK → [landing page](https://buffedlizard55-lab.github.io/GEMSDOE4/docs/index.html) and [executive summary](https://buffedlizard55-lab.github.io/GEMSDOE4/docs/executive_summary.html) both open with the in-browser submission builder**: click **Build submission.tif**, the browser writes the exact single-band float32 GeoTIFF the DrivenData dialog asks for, re-reads its own bytes, and hands over the download — no install, no GPU, nothing uploaded. The download carries a **unique file name** and a suggested **Note** so two builds can never be confused.
>
> **Exact submission instructions → [How to submit, step by step](https://buffedlizard55-lab.github.io/GEMSDOE4/docs/how_to_submit.html)** (a subpage of the executive summary): the file, its sha256 re-hashed at build time, six routes to produce it (A–F), the validation gate, the click-by-click upload path, and the fix for the platform's `Predicted values must be in range [0, 1]` rejection.

---

## 0. PROJECT CHARTER — read this first, every session

This section is the standing brief. It is the starting point for every work session on this
repository: read it, check the work against it, and only then write code.

### 0.0 The standing brief — the request this repository exists to satisfy

Recorded here so every session starts from the same list instead of from whoever remembers it last.
Each item carries its measured status; "done" always means *measured in this repository*, never
asserted.

| # | The ask | Status |
|---|---|---|
| 1 | Copy the entire repo and site from GEMSDOE (the site that scored 0.1563) into this repo, because more sites are being created for more submissions. | ✅ Done — the full GEMSDOE tree (413 files) is here; `buffedlizard55-lab/GEMSDOE` remains the source of record for the deep line. |
| 2 | Generate a **different, unique** submission — not `extradr19` (0.1563) — via a unique approach that can score **higher than 0.3049**. | ✅ Built and measured (sessions 32–33) — the "New-Fault-First" union now includes a **proxy_only** member (SGMC code-2 positives only, never a catalogue fault). Shipped sha `c1da7dd9…`, validator PASSED 10/10 here. Out-of-sample proxy DTI **0.1996** on the clean fold 1 and **0.247454** on the 17-block out-of-sample pool vs the prior union's 0.233098 (**+0.014356, P = 1.0, CI95 [+0.0080, +0.0202]**). Unscored on the real board (see §0.5) — the upload is the one remaining human step. |
| 3 | Put the full prompt in the README and read it every time work starts, so there is a strong base to keep improving something useful for everyday use — it must remove the need to check everything by hand and give an up-to-date current feed. | ✅ This section is that list. The site is the "current feed": it is rebuilt by CI from the measured evidence, so the numbers on the page are the numbers in the repository. |
| 4 | Keep the Arena core values — **Maximize P(Win)** and **Own the Outcome** — as the focal point when building, developing, researching, suggesting upgrades and implementing. | ✅ §0.4, and the reason every claim here is measured rather than argued. |
| 5 | Work line by line verifying from official verified trusted sources, with links for manual review. No manual input. Work autonomously. Flag irregularities for review. No hallucinations. Verify line by line. | ✅ §8 and `scripts/verify_rules_quotes.py` (29/29 quoted sentences exact-matched against the rules PDF). Session 33's line-by-line pass found four irregularities and fixed all four: the stale submission page on `main` (CI run 36209417841), the readiness record a weaker checkout could silently overwrite, `n_scoreable_blocks: 0` reported for the winning arm of the adoption contrast, and the fact that folds 2/3 are *in-sample* for the members (which is why the honest pool is folds 0+1, not 2+3). Each is written down in `STATUS.md` §Session 33 with the measurement that proved it. |
| 6 | The site must generate the submission TIF as easily as "download a file to click into the competition", obvious at the very beginning of the site / executive summary. | ✅ Both the landing page and the executive summary open with the in-browser builder; `scripts/check_site_generator.py` re-runs the browser's own pipeline headless and is gate 9 of the readiness check. |
| 7 | Fix the platform rejection `Predicted values must be in range [0, 1]`. | ✅ Root-caused (NaN inside the valid region, finite outside) and fixed by `scripts/sanitize_submission.py` + `conform_to_template()`; validator checks 16–17; two-sided evidence committed. |
| 8 | Provide a unique name and a short comment (e.g. "clustering with k=25") to tell submissions apart. | ✅ The download carries a unique per-build file name and the suggested Note `nff-po-drop-nff42 · proxy_only diversity · union k=2 of 5 (t0=0.18, w=0)`. |
| 9 | Create an executive-summary subpage explaining exactly how to make a submission into the contest. | ✅ `docs/how_to_submit.html`, a subpage of the executive summary. |
| 10 | Work the next steps from previous sessions first. | ✅ Each session opens with the previous session's open items; `SUGGESTIONS.md` and `LIMITATIONS.md` carry the queue. |
| 11 | Goal: top of the leaderboard. Understand the problem, collect all data, organise it into a clean, easily auditable table with official verified links. | ✅ §3 is that table (every row: source, link, licence, what it is used for); §1 is the problem summary. |
| 12 | Tell the user my limitations and what access is needed; use only free publicly available official/verified sources for third-party or external data. | ✅ §2 and `LIMITATIONS.md` — and the honest headline is that the score itself needs a DrivenData login this sandbox does not have. |
| 13 | Do own research (deep research, scientific literature research), organise knowledge for critical thinking, autonomously, constantly reviewed and improved; provide and implement suggestions. | ✅ `docs/literature.md`, `docs/references.md`, `docs/DISCOVERY_PLAN.md`; suggestions are implemented and re-measured in `SUGGESTIONS.md`. |
| 14 | Run through multiple passes: pass 1 implement completely + verify; pass 2 review for bugs/missing requirements/incorrect assumptions/edge cases and fix; pass 3 re-check the entire implementation against the original request and improve accuracy, reliability, completeness, code quality. | ✅ This table *is* the pass-3 re-check against the original request; pass 2 found the two real defects recorded in `STATUS.md` (chunk-boundary row duplication, unconformed 0.0 outside the footprint). |
| 15 | Create a pull request and merge it onto main; then make suggestions for remaining work and limitations blocking success, to be worked on next session. | ✅ PRs #1–#7 merged onto `main`; this session opens PR #8 (fix the red site-page gate on `main`, the two silent-overwrite defects, and the pooled adoption contrast). `SUGGESTIONS.md` §Session 33 and `LIMITATIONS.md` §Session 33 are the remaining-work and blocking-limitation lists. |
| 16 | GitHub Pages site: clean UI, user friendly, simple, organised; all relevant information easy to read, with official verified links as sources. | ✅ Live at <https://buffedlizard55-lab.github.io/GEMSDOE4/> (Pages status `built`); `scripts/audit_docs.py` re-checks every link on every run and PASSes. |

### 0.1 What we are trying to do

Place **top of the leaderboard** in the DOE GEMS Prize Challenge on DrivenData —
<https://www.drivendata.org/competitions/306/competition-doe-gems/> (problem description
<https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>, about page
<https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/>, data tab
<https://www.drivendata.org/competitions/306/competition-doe-gems/data/>). Prize pool
**$300,000** ($50k Phase 1, $250k Phase 2); the official rules are
<https://docs.nlr.gov/docs/fy26osti/96647.pdf> and the reference solution is
<https://github.com/drivendataorg/gems-prize-reference-solution>.

### 0.2 Where we stand (measured, 2026-09-26)

Five submissions have been scored on the public board so far (names as they appear in the
submission form's Note field):

| Note on the board | Repo / site | Strategy | Public score |
|---|---|---|---|
| `extradr19` | buffedlizard55-lab/GEMSDOE | 11-fold deep ensemble, `floor 0.1, thin, width 0` | **0.1563** |
| `smashi34` | buffedlizard55-lab/GEMSDOE2 | dual-family union | 0.1560 |
| `smrtdoog5` | buffedlizard55-lab/GEMSDOE3 | pindrop nodes | 0.1193 |
| `SDCF9` | buffedlizard55-lab/GEMSDOE3 | dense ridge control | 0.1152 |
| `wbg1` | buffedlizard55-lab/GEMSDOE3 | catalogue-gap target | 0.0830 |
| `nff-po-drop-nff42` | **this repo (GEMSDOE4)** | new-fault-first union, k = 2 of 5, `proxy_only` member | **not yet uploaded** |
| **best on the board** | — | — | **0.3049** |

Our best scored line is ~half the leading score. The task for this repository is therefore not
"ship another variant of the same model" but **a different strategy, researched and measured,
that can score above 0.3049**. The file this repository ships today
(`data/evidence/combined/submission.tif`, sha256 `c1da7dd9…`, 547,082 B, `validate_submission.py`
PASSED 10/10 in this sandbox) is that line's current candidate: the adopted **k = 2 of 5** union
(`deep11 ∪ classical ∪ nff43 ∪ nff45 ∪ po46`), where `po46` is the first member supervised on
**SGMC code-2 only** (never a catalogue fault).

Its evidence, on the SGMC proxy population (R = 3, α = 0.2, β = 0.8), against the 5-member union
it replaced (`19de9950…`):

| scope | ref | cand | Δ | blocks | P(cand>ref) |
|---|---:|---:|---:|---:|---:|
| fold 1 — clean (out-of-sample, never swept) | 0.189714 | 0.199605 | +0.009890 | 9 (COARSE) | 1.0 |
| **pooled folds 0+1 — out-of-sample pool** | **0.233098** | **0.247454** | **+0.014356** | **17 (adequate)** | **1.0** |
| pooled folds 2+3 — *in-sample for the members, upper bound only* | 0.177017 | 0.203354 | +0.026337 | 17 | 1.0 |

Session 33 added the pooled read (`--fold 0,1` in `scripts/paired_union_contrast.py`,
`data/evidence/union_po_loo/contrasts/drop_nff42_vs_committed_pooled01.json`): **17 resampling
units, above this repository's 12-block readable-CI bar, CI95 [+0.0080, +0.0202] excluding zero**.
Folds 0 and 1 are the only two folds `scripts/newfault_detector.py` excludes from the members'
training, so they are the only honest pool; fold 0 is also the fold the emission-policy sweep
selected on, so the pooled number sits between a clean read and a biased one, and folds 2/3 are
in-sample (+0.0263) and committed only as an upper bound. The transfer to the real leaderboard is
still untested until a human uploads it; every number here is the SGMC surrogate, not the scored
set. Suggested submission Note:
`nff-po-drop-nff42 · proxy_only diversity · union k=2 of 5 (t0=0.18, w=0)`.

### 0.3 The strategy this repository is built on (GEMSDOE4, "New-Fault-First")

1. **Score the population the rules actually score.** Verified verbatim from the rules PDF
   (`data/evidence/rules_quotes.json`): *"In Phase 1, submissions will be evaluated against a
   privately withheld subset of the original new fault dataset compiled by expert reviewers"*
   (§1.1) and *"Submissions will be reevaluated against the full, revised new fault dataset using
   the same distance-weighted Tversky index"* (§1.1). Both prize phases score the **new faults**;
   the public catalogue in `data/labels.tif` is the *training* set, never the test set. Every
   earlier line in this project tuned its emission policy on the catalogue population — this
   repository does not.
2. **Supervise with an independent fault compilation**, so the detector sees faults the catalogue
   does not contain (`scripts/newfault_detector.py`).
3. **Give the classifier lineament geometry** (multi-scale ridge-ness, structure-tensor coherence,
   windowed context) instead of raw pixel values (`src/lineament_features.py`).
4. **Combine structurally different detectors** (deep ensemble ∪ classical GBM ∪ lineament NFF)
   and pick the combination rule on held-out geography, on the new-fault population
   (`scripts/combine_newfault.py`).
5. **Select, then measure on a fold nothing touched**, with a pre-registered support window, so a
   number quoted in this repository is not an artefact of `max()` over a noisy grid.

### 0.4 Standing rules for all work here

- **Core values: Maximize P(Win) and Own the Outcome.** Weigh tradeoffs, assess risk, choose the
  path that maximises the probability of winning; own results end to end rather than waiting to be
  assigned the next slice.
- **Work line by line, verifying from official verified trusted sources, with links for manual
  review.** No manual input — research, deep research, scientific-literature research and
  organisation of that knowledge are done autonomously and re-reviewed.
- **No hallucinations.** Every number in this repository has a measurement behind it; every rule
  sentence is a verbatim quote checked against the rules PDF by `scripts/verify_rules_quotes.py`.
  **Flag irregularities for review rather than silently fixing them.**
- **The site must make a submission trivial.** One click to a valid `.tif` (or the `.zip` the dialog
  also accepts), obvious on the first screen, with a unique file name and a short Note for the
  submission form.
- **Run every task through multiple passes.** Pass 1 implement and verify; pass 2 review for bugs,
  missing requirements, wrong assumptions and edge cases; pass 3 re-check the whole implementation
  against the original request and improve accuracy, reliability, completeness and code quality.
- **Open a pull request and merge it**, then state plainly what work remains and which limitations
  block a successful project.

### 0.5 Known limitations of this repository (see `LIMITATIONS.md`)

- No DrivenData credentials here → no automatic download from the data tab and no automatic
  upload/leaderboard read (both are login-gated; verified).
- No GPU and no unrestricted egress → the 418 MB feature stack arrives through the sha256-pinned
  git bridge, the deep models were trained on GitHub runners, and every number quoted here is
  produced on CPU.
- The private expert labels do not exist locally → the new-fault population is measured through an
  independent public compilation used as a surrogate, and the public leaderboard remains the only
  unbiased test of that transfer.

---

# GEMSDOE — Geologic Enhanced Mapping System Prize Challenge

> **The file to submit, in one click → the published site leads with the in-browser submission builder** ([landing page](https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html), [executive summary](https://buffedlizard55-lab.github.io/GEMSDOE/docs/executive_summary.html)): click **Build submission.tif**, the browser writes the exact single-band float32 GeoTIFF the DrivenData dialog asks for, re-reads its own bytes, and hands over the download — no install, no GPU, nothing uploaded (added 2026-09-24).
>
> **Start here → [Executive summary: How to submit](https://buffedlizard55-lab.github.io/GEMSDOE/docs/executive_summary.html)** (markdown: [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md))
>
> **Subpage of the executive summary → [How to submit, exactly](https://buffedlizard55-lab.github.io/GEMSDOE/docs/how_to_submit.html)** — the recipe: the file, its sha256 (re-hashed at build time), **the same browser generator** (that page's §3), six routes to produce it (A–F, from clicking Build on the page to the CPU-only workflow), the validation gate, the click-by-click upload path, and a **gate table measured in this checkout** (`data/evidence/submission_readiness.json`) in which the human-only steps are labelled `HUMAN` rather than counted as done.

### Top-Leaderboard Solution Framework

**Competition:** [GEMS Prize Challenge on DrivenData](https://www.drivendata.org/competitions/306/competition-doe-gems/)  
**Problem Description:** https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/  
**About / Resources:** https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/  
**Data Tab (requires login):** https://www.drivendata.org/competitions/306/competition-doe-gems/data/  
**Official Rules (HeroX):** https://www.herox.com/GEMSPrize/resource/2274 → PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf  
**Reference Solution:** https://github.com/drivendataorg/gems-prize-reference-solution  
**Prize:** $300,000 total ($50k initial, $250k final) | End Date: Dec 3, 2026 11:59pm UTC  
**Sponsor:** DOE Office of Geothermal + National Lab of the Rockies

---

## 1. Problem Summary (verified, no hallucinations)

**Task:** Develop models that predict presence of geological faults (fractures indicative of geothermal resources) from geophysical data in the **GeoDAWN region** (Geoscience Data Acquisition for Western Nevada).

- **GeoDAWN** = high-resolution airborne magnetic + radiometric surveys in NW Nevada & E California, covering Walker Lane and western Great Basin. Source: USGS Data Release [https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7), DOI [https://doi.org/10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ), and overview [https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)

- **Labels:** USGS Quaternary Fault and Fold Database + INGENIOUS project. Known faults are incomplete; challenge provides new expert-labeled hidden faults for test. Final scoring uses expanded label set after expert review of all submissions.

- **Provided Features (training_features.tif, UTM 11N EPSG:32611, 100m resolution):**
  - Surface conductivity & depth to conductive base
  - Detrended elevation & slope of detrended elevation
  - Dilatation rate, shear strain rate, second invariant of strain rate tensor
  - Isostatic gravity anomaly & slope of isostatic gravity anomaly
  - Magnetics: reduced-to-pole magnetic anomaly, total magnetic intensity, vertical & horizontal slope of TMI, top-of-crustal magnetic source depth estimate
  - Density of earthquakes
  - Plus CSV `1m_DEM_links.csv` for high-res DEM download

- **Metric:** Distance-weighted Tversky index with triangular kernel R=300m (3 pixels), alpha=0.2 (FP penalty low), beta=0.8 (FN penalty high). Rewards high recall, tolerates small misalignment.

  Formulas (from problem page):
  - k(d) = max(1 - d/R, 0)
  - TP_w = Σ_{g∈G} max_{x:d(x,g)≤R} p(x) k(d(x,g))
  - FP_w = Σ_{x:p(x)>0} p(x) [1 - max_{g∈G} k(d(x,g))]
  - FN_w = Σ_{g∈G} [1 - max_{x:d(x,g)≤R} p(x) k(d(x,g))]
  - DTI = TP_w / (TP_w + α FP_w + β FN_w + ε)

- **Submission:** Single-layer GeoTIFF float32 in [0,1], same CRS (EPSG:32611), same resolution (100m), same bounds as training, NaN outside.

- **Competition Structure:** Same submission scored twice:
  - Initial Round: private pre-labeled hidden faults (public leaderboard = public subset)
  - Expert review uses all submissions to find candidate new faults → expanded label set
  - Final Round: rescore against expanded set → true discoveries rewarded.

---

## 2. Limitations & Required Access (as requested)

### Session update 2026-09-12 (late) — verified in a *fresh* sandbox
- Environment rebuilt from PyPI inside the sandbox (numpy/rasterio/scikit-image/scipy/torch/smp/torchvision/geopandas all import; exact pins in `requirements.verified.txt`).
- **Real public data can enter the sandbox** through the GitHub route (only `github.com`, `codeload.github.com`, `api.github.com`, `pypi.org`, `files.pythonhosted.org` pass the egress allowlist): the GeoDAWN 22103 area-1 rasters + INGENIOUS QFaults + INGENIOUS seismicity were fetched from a pinned commit of `jklinck/geothermal_research` and the full train → inference → validate → score pipeline was executed on them.
- Everything quantitative in this repo now has a measurement behind it: **`docs/results.html`** (metric == brute-force transcription of the official formulas, 19-test suite, shaping experiment, 3-arm loss A/B). Fabricated "expected DTI" tables were **deleted**.
- New: differentiable transcription of the competition metric as the training loss (`src/losses.py`), metric-derived submission shaping (`src/submission_optim.py`), memory-safe full-raster scorer, leak-free patching, persisted normalisation stats, manifest-driven inference.

### Current Limitations in this Sandbox
- **DrivenData authentication — still needed for SUBMITTING, no longer for data.** The official
  rasters are in `data/` via the sha256-pinned git bridge (`data/bridge/` →
  `python scripts/assemble_data_bridge.py`). What still needs an account + enrollment: uploading
  submissions and reading the leaderboard (3 submissions/week, rules §3.2).
- **No GPU / limited CPU:** Training large segmentation models (U-Net, SegFormer) ideally needs GPU (CUDA or Apple MPS). Sandbox is CPU-only (2 vCPU, 3.9 GB RAM); the full 19-band pipeline fits after the session-9 memory fixes, but the leaderboard config needs a GPU.
- **No large external data pre-downloaded:** GeoDAWN grids are GB-scale, 1m DEM tiles are many GB. We provide download scripts but cannot bulk-download here.
- **Private test labels unavailable:** Expected; we must use cross-validation and visual inspection.

### What We Need for Full Competitive Run
1. **DrivenData account + competition enrollment** to upload submissions and observe the leaderboard (data is no longer blocked).
2. **Compute:** GPU machine (e.g., 1x A100, 24GB+ VRAM) or multi-GPU for ensemble training. Reference solution recommends CUDA 12.6 or 13.0.
3. **Storage:** ~50GB for GeoDAWN + DEM + INGENIOUS + processed patches.
4. **Optional:** Access to USGS AWS Open Data for 3DEP lidar: https://registry.opendata.aws/usgs-lidar/
5. **Time:** ~5-10h training for full ensemble (MC=10, 50 epochs).

### What Is Allowed for External Data
Per rules: any data with license permitting use in challenge and sharing with sponsor for evaluation. All our listed external sources are **public domain (U.S. Government)** or CC-licensed.

---

## 3. Verified External Data Catalog (Auditable, No Hallucinations)

All entries have official verified links for manual review. See also `docs/data_catalog.csv` and `docs/data_catalog.json`.

| # | Dataset Name | Official Source | Verified Link(s) | License | Relevance | How Used |
|---|--------------|-----------------|------------------|---------|-----------|----------|
| 1 | **GeoDAWN Airborne Magnetic & Radiometric Surveys** | USGS | ScienceBase: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 <br> DOI: https://doi.org/10.5066/P93LGLVQ <br> USGS overview: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and | Public Domain (USGS) | Primary geophysics in challenge | Provides high-res mag & radiometric; we use to augment training_features |
| 2 | **USGS Quaternary Fault & Fold Database** | USGS Earthquake Hazards | Interactive: https://www.usgs.gov/programs/earthquake-hazards/faults <br> ScienceBase: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 <br> DOI: https://doi.org/10.5066/P9BCVRCK <br> ArcGIS: https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer | Public Domain | Ground truth labels | Training labels source; we verify alignment, use for additional supervision |
| 3 | **USGS 3DEP 1m DEM / Lidar** | USGS National Geospatial Program | About: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services <br> Downloader: https://apps.nationalmap.gov/downloader/ <br> LidarExplorer: https://apps.nationalmap.gov/lidar-explorer/ <br> AWS: https://registry.opendata.aws/usgs-lidar/ | Public Domain ("free of charge and without use restrictions") | Elevation detail | Official source for `1m_DEM_links.csv`; we compute slope, curvature, TRI, TPI, hillshade |
| 4 | **INGENIOUS Great Basin Regional Dataset Compilation** | GBCGE / DOE GDR | GDR: https://gdr.openei.org/submissions/1391 <br> DOI: https://doi.org/10.15121/1881483 <br> OSTI: https://www.osti.gov/biblio/1881483 <br> Project: https://gbcge.org/current-projects/ingenious/ | **CC BY 4.0** (verified on GDR page) | Features in challenge (conductivity, strain, gravity, etc.); also origin of training labels per rules PDF §3.3 | Source for conductivity, strain rate, gravity, earthquake density layers; sub-datasets have own USGS DOIs (P9TWT2LU MT conductance, P9MQRCBY detrended elevation, P9Z6SA1Z gravity/magnetics, P9BZPVUC heat flow, P9YL58W6 slip/dilation tendency) |
| 5 | **GBCGE Subsurface Database Explorer** | GBCGE | DOI: https://doi.org/10.15121/1987556 <br> OSTI: https://www.osti.gov/dataexplorer/biblio/dataset/1987556 | Public | Geothermal wells, springs, temps | External features: temp probes, well chemistry |
| 6 | **Geothermal Favorability – INGENIOUS Features** | USGS + INGENIOUS | ScienceBase: https://www.sciencebase.gov/catalog/item/66e88690d34e0606a9db9b43 | Public | ML favorability | 16 input features curated for publication |
| 7 | **USGS EarthMRI** | USGS Mineral Resources | Portal: https://mrdata.usgs.gov/earthmri/ <br> Overview FS: https://pubs.usgs.gov/publication/fs20203055 | Public Domain | Critical mineral context | Understanding acquisition areas |
| 8 | **INGENIOUS Papers & Methods** | Academia | Mattéo et al 2021: https://doi.org/10.1029/2020JB021269 <br> Hermant et al 2025: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf | Open Access | Method inspiration | Automatic fault mapping with deep learning |

**No hallucinations:** Every link above was verified via web_search and fetch_page during development. See `docs/references.md` for fetch logs.

**Irregularities flagged:**
- Competition data tab requires login – cannot verify exact file sizes/names without account. Reference solution mentions `numeric_features.tif` vs `training_features.tif` naming drift – we handle both.
- 1m DEM links CSV likely points to The National Map – we provide fallback via LidarExplorer.
- Some INGENIOUS layers (e.g., detrended elevation) derivation not fully documented – we re-derive from 3DEP.

---

## 4. Solution Strategy to Place Top of Leaderboard

### Why Reference Solution is Insufficient
- Single U-Net, 5 MC splits, 5 epochs, basic normalization, no external DEM, no post-processing, no TTA, no ensemble blending, no distance-weighted loss approximation.

### Our Improvements (implemented in `src/`)

#### A. Data Engineering
- **Normalization (`src/dataset.py:fit_norm_stats/apply_norm_stats`):** per channel, NaN-aware 1–99 percentile
  clip → robust z-score → [0,1]. The reference solution's plain min–max over the whole raster (its cell 5) is
  not robust to the heavy-tailed magnetic/gravity bands; `norm_mode: minmax` stays available for A/B. Statistics
  are written to `outputs/norm_stats.json` at train time and re-loaded at inference, so reproduction does not
  depend on chance (rules §3.5).
- **External DEM pipeline (`src/external_data.py` + `scripts/download_dem_tiles.py`):** resample 1 m 3DEP tiles to
  100 m and derive slope, aspect, profile/plan curvature, TPI, TRI, hillshade, detrended elevation. Code and
  config are in place; the tile hosts are unreachable from this sandbox, so this is the one improvement in this
  list that is **not yet measured** — flagged rather than claimed.
- **Patching (`make_patches`):** test windows are chosen first and zeroed in the global raster *before* training
  windows are cut (no CV leakage; regression-tested); train windows need ≥3 fault pixels, and `neg_fraction`
  (0.35) of empty windows is kept so the model learns the background — this is deliberate empty-window sampling,
  not mined hard negatives, and the repo says so. Patch 256 with `train_step=128` (2× overlap) vs the
  reference's non-overlapping 128 — km-scale lineaments need the wider context.
- **Augmentation (`FaultDataset`, label-consistent numpy):** random crop (scale 0.75–1.0, re-padded rather than
  resized because a 1-px-wide trace is destroyed by resampling), horizontal flip, vertical flip, k×90° rotation,
  Gaussian noise σ=0.01 — applied identically to features, labels and the FP-weight map. The rotation set is
  exactly the set our TTA inverts. (No elastic warp or brightness/contrast jitter: those were listed here before
  they were implemented, and photometric jitter on already-normalised geophysics is of doubtful value.)

#### B. Model Architecture
- **Encoders:** EfficientNet-B3/B5, ResNet-50, MIT-B2 (SegFormer) pretrained on ImageNet.
- **Decoders:** UNet++, DeepLabV3+, SegFormer, UPerNet.
- **Loss:** `0.3·BCE + 0.5·(1 − DTI_surrogate) + 0.2·focal-Tversky`. The DW term is a differentiable
  transcription of the official metric (29-offset triangular kernel, R=3 px, α=0.2, β=0.8) and is asserted
  equal to `1 − DTI` of the scorer in the test suite — the reference solution's pixel-wise Tversky actively
  fights the R=300 m tolerance the committee put into scoring.
- **Submission shaping (measured, not guessed):** FP<sub>w</sub> is a *sum over area* while |G| counts only fault
  pixels, and TP<sub>w</sub> is a *max over the R-neighbourhood* → hard floor (zero, don't shrink) +
  distance-R dominating thinning, tuned by pooled held-out search and stored in `outputs/manifest.json`.
- **Training:** 10 MC splits (70/30 by test *windows*, not stratified k-fold), AdamW init lr 1e-4 with
  OneCycleLR (peak 10×), batch 16, patch 256, AMP on CUDA, early-stopping patience 10, and **model selection on
  shaped held-out DTI** (the quantity scored), logged per epoch as `DTI_raw=` and `DTI_shaped=`.
- **Inference:** sliding window at `overlap: 0.75` with Gaussian blending, 8-way TTA (4 rotations × flip, each
  inverted exactly), ensemble over splits × architectures with optional DTI-softmax weights; the submission stays a
  probability raster (`threshold: null`) because the metric consumes soft values.

#### C. Post-processing for the final round (discover new faults)
- **Shaping, tuned not guessed:** floor `t0` + distance-R dominating thinning, searched jointly on pooled
  held-out windows during training and stored in `outputs/manifest.json` (`src/submission_optim.py`); measured
  2.8× on a CPU smoke model (`docs/results.html` §3).
- **Continuity (config `postprocess`, all optional and off-by-default where unproven):** Frangi vesselness
  enhancement (weight 0.35, scales 1–10), morphological closing (3×3), `min_fault_length: 5` px connected-component
  filter. Skeletonization is `false` until it is validated on held-out windows — a skeleton can *lose* DTI when the
  label is a thick band, which is a measured result (`test_thinning_wide_band_onto_a_narrow_fault_raises_dti`), not
  an opinion.
- **Deliberately not implemented:** a Hough-transform line detector and slope/curvature masks to veto false
  positives. Both are plausible, neither is verified against this metric, and rules §3.2 asks for code we can
  defend; they are recorded as ideas in `SUGGESTIONS.md` instead of shipped as claims.

#### D. Metric Implementation
- Exact reproduction of distance-weighted Tversky with triangular kernel R=300m in `src/metrics.py`, using scipy distance transforms for efficiency.
- Used for validation and early stopping.

#### E. External Data Fusion (allowed)
- All external data from table above, public domain.
- We never use private test labels.

### Measured gains (not a leaderboard prediction)
- Reference-style objective (plain pixel-wise Tversky, the loss its notebook actually uses): measured **0.0659**
  shaped held-out DTI in our 3-arm comparison, vs 0.0913 for the metric-aligned objective — same data, seeds,
  budget and post-processing, so the gap is attributable to the objective, not to the harness. What we cannot
  say from that is anything about absolute leaderboard position: the reference model was never run on the official
  rasters here (its data is login-gated), and our runs use `pretrained: false` because the weight host is blocked.
- **We publish no expected leaderboard DTI** — there is no public history for this new competition, so such a
  number would be invented. What we publish is measured on real public data and reproducible
  (`docs/results.html`): submission shaping alone moved held-out DTI from **0.0437 → 0.1210** on a 4-epoch CPU
  smoke model, and a 3-arm loss comparison gave 0.0913 (metric-aligned combined) / 0.0838 (pure DW-Tversky) /
  0.0659 (the reference's plain Tversky). The public leaderboard is the first unbiased signal
  (3 submissions/week max — rules PDF §3.4).

---

## 5. Repo Structure

```
GEMSDOE/
├── README.md (this file)
├── docs/ (GitHub Pages site)
│   ├── index.html (clean UI)
│   ├── style.css
│   ├── data_catalog.csv
│   ├── data_catalog.json
│   ├── references.md
│   └── assets/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── inference.py
│   ├── postprocess.py
│   ├── submission_optim.py   # metric-derived floor + dominating-set thinning
│   └── external_data.py
├── tests/test_metric.py      # 20 falsifiable checks of every claim in this repo
├── configs/
│   ├── config.yaml           # leaderboard config (GPU + official data)
│   └── config_recon_cpu.yaml # CPU smoke config on reconstructed public data
├── scripts/
│   ├── audit_docs.py                  # documentation audit gate (runs in CI before deploy)
│   ├── build_reconstruction_dataset.py# public-source -> GeoTIFF stack for pipeline verification
│   ├── build_dem_links.py             # DEM tile list from the competition links file
│   ├── download_competition_data.sh   # run where DrivenData login works
│   ├── download_dem_tiles.py          # 3DEP 1 m tiles (S3 listing or the links file)
│   ├── download_external.sh           # INGENIOUS / QFaults subsets
│   ├── fetch_dem_links_pdf.py         # regenerate data/dem_links.json + evidence
│   ├── generate_dummy_submission.py   # format-valid placeholder for pipeline tests
│   ├── measure_submission_variants.py  # the shaping table in docs/results.html
│   ├── prepare_data.py                # pre-flight CRS/resolution/bounds checks (exit != 0 on failure)
│   ├── run_ab_loss_experiment.sh      # loss A/B used for the docs table
│   └── validate_submission.py         # submission format validator (exit 1 = broken)
├── requirements.txt            # runtime/model ranges
├── requirements-dev.txt        # runtime + pytest/PDF audit dependencies
├── requirements.verified.txt   # exact versions installed + import-verified
├── environment.yml
├── docs/                       # GitHub Pages site (index, methodology, data, results, ...)
└── .github/workflows/pages.yml # audit gate -> deploy
```

---

## 5b. What actually runs today (measured, 2026-09-12)

```bash
pip install -r requirements-dev.txt       # runtime + pytest/PDF audit dependencies
python src/metrics.py --self-test          # 8 checks: scorer == literal formula transcription
python -m pytest tests -q                   # full regression suite (optional text extract test may skip)
python scripts/build_reconstruction_dataset.py        # needs the pinned public-source tree (see data/README.md)
python -m src.train     --config configs/config_recon_cpu.yaml
python -m src.inference --config configs/config_recon_cpu.yaml --out outputs_recon/submission.tif
python scripts/validate_submission.py --pred outputs_recon/submission.tif \
       --sample data/reconstructed/recon_sample_submission.tif \
       --train data/reconstructed/recon_training_features.tif     # exit 0 = format-valid
python scripts/measure_submission_variants.py --config configs/config_recon_cpu.yaml
bash scripts/run_ab_loss_experiment.sh
```

## 6. Quickstart (once you have DrivenData data)

```bash
# 1. Clone
git clone https://github.com/buffedlizard55-lab/GEMSDOE.git
cd GEMSDOE

# 2. Env (conda or pip)
conda env create -f environment.yml
conda activate gemsdoe
# or
pip install -r requirements-dev.txt

# 3. Place competition data
mkdir -p data/
# Download from https://www.drivendata.org/competitions/306/competition-doe-gems/data/
# Expected files:
# - training_features.tif (or numeric_features.tif)
# - labels.tif (or faults.tif)
# - sample_submission.tif
# - 1m_DEM_links.csv

# 4. Optional external data
bash scripts/download_external.sh  # downloads INGENIOUS sample, QFaults shapefile

# 5. Train
python -m src.train --config configs/config.yaml

# 6. Inference -> submission.tif
python -m src.inference --config configs/config.yaml --model-dir outputs/best --out submission.tif

# 7. Evaluate (if you have val labels)
python -m src.metrics --pred submission.tif --true data/labels.tif
```

---

## 7. GitHub Pages

We provide a clean, user-friendly site in `docs/`:

- Deploy: Settings → Pages → Source: GitHub Actions (workflow in `.github/workflows/pages.yml`)
- Local preview: `cd docs && python -m http.server 8000`
- Site includes:
  - Problem overview
  - Metric visualization
  - Auditable data catalog with official links
  - Model architecture diagram
  - Limitations & needed access
  - References

URL after deploy: `https://buffedlizard55-lab.github.io/GEMSDOE/`

---

## 8. Verification & No Hallucinations Statement

- All external links were fetched via `fetch_page` or `web_search` during development.
- Data catalog entries have DOIs or USGS official pages.
- Code for metric matches problem description formulas exactly.
- No synthetic fault data invented.
- Flagged irregularities in section 3.

---

## 9. License & Citation

- Code: MIT (as reference solution)
- Data: Respective licenses (USGS Public Domain, INGENIOUS CC)
- If using this repo, cite:
  - Glen & Earney 2024 GeoDAWN DOI 10.5066/P93LGLVQ
  - USGS QFaults DOI 10.5066/P9BCVRCK
  - INGENIOUS DOI 10.15121/1881483
  - Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/

---

## 10. Contact / Next Steps

- TODO for owner: Add DrivenData API token to download data automatically (if allowed).
- TODO: Run full training on GPU and submit to leaderboard.
- TODO: Expert review contribution – our predictions aim to maximize true new fault discovery for Final Round.

Built with ❤️ for geothermal discovery.
