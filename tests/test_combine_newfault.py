"""End-to-end test of scripts/combine_newfault.py on synthetic detector fields.

The combiner is where the GEMSDOE4 strategy becomes a file, so what it must be prevented from
doing is exactly what a "union" could do wrong:

* drop the template's NaN mask (the platform rejection this repo already suffered) - the members
  are read through nan_to_num, so the union is finite *outside* the survey footprint by
  construction and must be masked back;
* select a combination the report does not actually rank on;
* ignore its own pre-registered support window;
* quote a measurement taken on the geography the sweep scored.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "combine_newfault.py"


def _mod():
    spec = importlib.util.spec_from_file_location("combine_newfault", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(path: Path, arr: np.ndarray, dtype: str, nodata=None) -> Path:
    profile = dict(driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                   count=1 if arr.ndim == 2 else arr.shape[0], dtype=dtype, crs="EPSG:32611",
                   transform=from_origin(500000.0, 4100000.0, 100.0, 100.0), nodata=nodata,
                   compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype, copy=False), 1 if arr.ndim == 2 else list(range(1, arr.shape[0] + 1)))
    return path


def _fixture(d: Path, size: int = 192):
    """Three detectors that each see a different trace, plus a proxy-only trace none of them knows.

    A (binary) is right about the catalogue trace at column 30 and blind to column 150.
    B (binary) is right about column 150 and blind to column 30.
    C (probability) is smooth and moderately confident about both, plus a bit of noise elsewhere.
    The proxy truth contains BOTH traces - the population the combiner is supposed to target.
    """
    rng = np.random.default_rng(13)
    inside = np.zeros((size, size), bool)
    inside[4:size - 4, :] = True

    def band(col, halfwidth=2, rows=(10, size - 10)):
        m = np.zeros((size, size), bool)
        m[rows[0]:rows[1], max(0, col - halfwidth):col + halfwidth + 1] = True
        return m & inside

    trace30, trace150 = band(30), band(150)
    # a long horizontal proxy-only trace: it crosses every 64-px block column, so whichever blocks
    # the balanced partition deals to the selection and measurement folds, both folds contain
    # proxy truth (a fold with no target truth cannot select or measure anything)
    trace_h = np.zeros((size, size), bool)
    trace_h[90:93, :] = True
    trace_h &= inside
    labels = np.full((size, size), -1, np.int8)
    labels[inside] = 0
    labels[trace30] = 1

    A = (trace30 & inside).astype(np.float32)
    B = (trace150 & inside).astype(np.float32)
    C = (0.65 * trace30.astype(np.float32) + 0.7 * trace150.astype(np.float32)
         + 0.8 * trace_h.astype(np.float32)
         + 0.15 * rng.random((size, size))).astype(np.float32)
    C = np.where(inside, C, np.nan).astype(np.float32)

    sample = np.where(inside, 0.0, np.nan).astype(np.float32)
    proxy = np.zeros((size, size), np.uint8)
    proxy[trace150] = 2                       # new-fault-like: not in the catalogue
    proxy[trace_h] = 2
    proxy[trace30] = 1
    return dict(
        a=_write(d / "a.tif", A, "float32", nodata=np.nan),
        b=_write(d / "b.tif", B, "float32", nodata=np.nan),
        c=_write(d / "c.tif", C, "float32", nodata=np.nan),
        labels=_write(d / "labels.tif", labels, "int8", nodata=-1),
        template=_write(d / "sample.tif", sample, "float32", nodata=np.nan),
        proxy=_write(d / "proxy.tif", proxy, "uint8"),
        inside=inside, trace30=trace30, trace150=trace150, trace_h=trace_h)


def _folds_with_truth(fx, want: str) -> list[int]:
    """The folds of this fixture's own partition that actually contain `want` truth.

    A fold with no target truth can neither select nor measure a combination, so the fixture picks
    the (selection, measurement) pair from what the partition really holds instead of assuming
    folds 0 and 1 are usable.
    """
    from src.blocks import assign_folds, block_table, scored_mask
    with rasterio.open(fx["labels"]) as src:
        lab = src.read(1)
    with rasterio.open(fx["template"]) as src:
        tmpl = src.read(1)
    with rasterio.open(fx["proxy"]) as src:
        pc = src.read(1)
    valid = np.isfinite(tmpl)
    truth = ((lab == 1) if want == "catalogue" else (pc == 2)) & valid
    table = block_table(lab.shape, 64, valid=valid, labels=(lab == 1))
    fold_of = assign_folds(table, 4, 42, "balanced")
    return [f for f in range(4) if int(truth[scored_mask(lab.shape, 64, fold_of, f)].sum()) > 0]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("combine")
    fx = _fixture(d)
    out_dir = d / "out"
    mod = _mod()
    proxy_folds = _folds_with_truth(fx, "proxy")
    assert len(proxy_folds) >= 2, "the fixture must give two folds proxy truth to select and measure"
    sel_fold, meas_fold = proxy_folds[0], proxy_folds[1]
    rc = mod.main(["--labels", str(fx["labels"]), "--template", str(fx["template"]),
                   "--proxy", str(fx["proxy"]), "--out-dir", str(out_dir),
                   "--member", f"a={fx['a']}", "--member", f"b={fx['b']}",
                   "--member", f"c={fx['c']}:prob",
                   "--block-px", "64", "--folds", "4", "--fold", str(sel_fold),
                   "--eval-fold", str(meas_fold),
                   "--floors", "7", "--dilates", "0", "--votes", "1",
                   "--max-emitted-fraction", "0.30", "--name", "synthetic-union"])
    report = json.loads((out_dir / "report.json").read_text())
    return dict(rc=rc, out_dir=out_dir, fx=fx, report=report, mod=mod)


# --------------------------------------------------------------------------------------- format
def test_main_rc_and_the_artifact_is_template_conformant(run):
    assert run["rc"] == 0
    sub = run["out_dir"] / "submission.tif"
    assert sub.exists()
    with rasterio.open(sub) as src:
        with rasterio.open(run["fx"]["template"]) as tmpl:
            assert (src.width, src.height, src.count) == (tmpl.width, tmpl.height, 1)
            assert src.crs == tmpl.crs and src.transform == tmpl.transform
            assert src.dtypes[0] == "float32"
            a, t = src.read(1), tmpl.read(1)
    finite = np.isfinite(a)
    assert np.array_equal(finite, np.isfinite(t)), \
        "the members are read through nan_to_num, so the union is finite outside the footprint " \
        "until conform_to_template masks it - failing this is the platform rejection"
    assert float(a[finite].min()) >= 0.0 and float(a[finite].max()) <= 1.0
    assert int(np.count_nonzero(a[finite])) > 0


def test_the_hash_sidecar_follows_the_bytes(run):
    """REGRESSION (session 34).  A stale `submission.sha256` next to a freshly written artifact is a
    a record of bytes that no longer exist: `scripts/package_submission.py`,
    `scripts/check_submission_readiness.py` and the packaging test all read that file as the
    artifact's identity.  The writer is the only place that knows the hash, so it must update it.
    """
    sidecar = run["out_dir"] / "submission.sha256"
    assert sidecar.exists(), "the combiner must leave a hash sidecar beside the artifact"
    recorded = sidecar.read_text().split()[0]
    live = hashlib.sha256((run["out_dir"] / "submission.tif").read_bytes()).hexdigest()
    assert recorded == live, "the sidecar must name the bytes on disk, not a previous run's"


def test_the_conformance_change_is_recorded_two_sided(run):
    ev = json.loads((run["out_dir"] / "sanitize.json").read_text())
    live = hashlib.sha256((run["out_dir"] / "submission.tif").read_bytes()).hexdigest()
    assert ev["after"]["sha256"] == live, "the evidence must describe the bytes on disk"
    assert ev["before"]["sha256_pixels"] != ev["after"]["sha256_pixels"]
    assert ev["changes"]["masked_outside"] > 0, "the union really is finite outside the footprint"
    assert ev["changes"]["filled_inside"] == 0


def test_the_report_quotes_the_bytes_on_disk(run):
    rep = run["report"]["submission"]
    assert rep["sha256"] == hashlib.sha256(Path(rep["path"]).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------- selection
def test_the_union_beats_its_weakest_member_on_the_target_population(run):
    """The metric algebra says a union of differently-biased detectors can beat each of them."""
    rep = run["report"]
    scores = rep["submission"]["global_scores"]
    assert scores["proxy"] > 0.0
    assert rep["submission"]["nonzero_px"] > 0
    # both synthetic traces are proxy truth; a union that misses one of them is not a union
    assert scores["proxy"] > 0.5, f"the union should cover both traces, got {scores['proxy']}"


def test_the_winner_is_the_argmax_over_eligible_rows_only(run):
    sel = run["report"]["selection"]
    assert sel["population"].startswith("proxy")
    eligible = [r for r in sel["candidates"] if r["eligible"]]
    assert eligible
    assert sel["winner"] in eligible
    key = sel["selection_key"]
    assert key == "proxy_dti_selection", \
        "the selection key must be the SELECTION fold's scope, not the whole grid"
    assert sel["winner"][key] == max(r[key] for r in eligible), \
        "the winner must be the argmax of the statistic the report says it ranked on"
    for row in sel["candidates"]:
        if not row["eligible"]:
            assert "cap" in row["eligibility_reason"] or "floor" in row["eligibility_reason"]


def test_the_selection_key_is_the_selection_fold_and_not_the_whole_grid(run):
    """REGRESSION (session 34).  Every candidate is scored on four scopes, and only ONE of them may
    pick the policy.  Until this session the picker read `proxy_dti` - the WHOLE-GRID score - while
    the report and the docstring both claimed the selection fold.  The whole grid is not a noisier
    version of the selection fold: the NFF members train on everything except folds 0 and 1, so
    folds 2/3 are in-sample and a whole-grid score is partly a measurement of memorisation.
    """
    sel = run["report"]["selection"]
    rows = sel["candidates"]
    for r in rows:
        for k in ("proxy_dti_selection", "proxy_dti_measurement", "proxy_dti_pooled01",
                  "proxy_dti_whole", "proxy_dti"):
            assert k in r, f"every candidate must carry {k} so each scope is auditable"
        assert r["proxy_dti"] == r["proxy_dti_whole"], \
            "the back-compatible name must still be the whole-grid number, labelled as context"
    assert set(sel["scopes_explained"]) >= {"proxy_dti_selection", "proxy_dti_measurement",
                                            "proxy_dti_pooled01", "proxy_dti_whole"}
    # the module's own rule, exercised directly on rows where the two keys disagree
    mod = run["mod"]
    disagreeing = [dict(proxy_dti_selection=0.10, proxy_dti=0.30, dilate=0, vote=1, eligible=True),
                   dict(proxy_dti_selection=0.20, proxy_dti=0.25, dilate=0, vote=2, eligible=True)]
    picked = mod.select_winner(disagreeing)
    assert picked["proxy_dti_selection"] == 0.20 and picked["vote"] == 2, \
        "the picker followed the whole-grid score to the row with the higher proxy_dti"
    assert mod.SELECTION_KEY == "proxy_dti_selection"


def test_an_unmeasurable_selection_scope_is_refused_not_scored_zero(run):
    """A fold with no truth pixels has no DTI.  `None` must stop the run: silently reading it as 0
    would rank a policy on a scope it cannot be measured on."""
    mod = run["mod"]
    with pytest.raises(SystemExit):
        mod.select_winner([dict(proxy_dti_selection=None, proxy_dti=0.9, dilate=0, vote=1,
                                eligible=True)])
    with pytest.raises(SystemExit):
        mod.select_winner([])


def test_the_support_window_can_exclude_a_candidate(run, tmp_path):
    """A cap of 0 must reject every candidate: the window is a gate, not a label."""
    fx = run["fx"]
    mod = _mod()
    out = tmp_path / "capped"
    with pytest.raises(SystemExit):
        mod.main(["--labels", str(fx["labels"]), "--template", str(fx["template"]),
                  "--proxy", str(fx["proxy"]), "--out-dir", str(out),
                  "--member", f"a={fx['a']}", "--member", f"c={fx['c']}:prob",
                  "--block-px", "64", "--folds", "4", "--floors", "3",
                  "--max-emitted-fraction", "0.0"])
    assert not (out / "submission.tif").exists(), "a refused run must not leave a file behind"


def test_the_measurement_fold_is_not_the_selection_fold(run):
    rep = run["report"]
    assert rep["selection"]["fold"] != rep["measurement"]["fold"]
    assert rep["measurement"]["scope"].startswith("blocks of fold")
    for pop, m in rep["measurement"]["by_scope"]["measurement"].items():
        assert m["n_gt"] > 0, f"the measurement fold must contain {pop} truth"


def test_the_vote_axis_spans_k_1_to_n_and_refuses_k_above_n(run, tmp_path):
    """Session 34: the search only ever swept k = 1 and k = 2, and the k = 3 family is the one that
    generalises best out of sample on the real grid.  A search that cannot express k = 3 cannot find
    it, so the default now spans k = 1..5 - and an impossible k must be refused by name rather than
    silently emitting an empty field."""
    fx, mod = run["fx"], run["mod"]
    out = tmp_path / "votes"
    mod.main(["--labels", str(fx["labels"]), "--template", str(fx["template"]),
              "--proxy", str(fx["proxy"]), "--out-dir", str(out),
              "--member", f"a={fx['a']}", "--member", f"b={fx['b']}",
              "--member", f"c={fx['c']}:prob",
              "--block-px", "64", "--folds", "4", "--floors", "3", "--dilates", "0",
              "--votes", "1,2,3,4,5", "--max-emitted-fraction", "0.30"])
    rep = json.loads((out / "report.json").read_text())
    votes = sorted({r["vote"] for r in rep["selection"]["candidates"]})
    assert votes == [1, 2, 3], f"3 members admit k <= 3, got {votes}"
    assert rep["selection"]["k_votes_swept"] == [1, 2, 3, 4, 5], \
        "the report must record what was ASKED for as well as what was runnable"
    assert rep["selection"]["n_members"] == 3


def test_member_provenance_is_carried_not_invented(run):
    members = {m["name"]: m for m in run["report"]["members"]}
    assert set(members) == {"a", "b", "c"}
    assert members["c"]["probability"] is True and members["a"]["probability"] is False
    assert all(m["sha256"] for m in members.values()), "each member must be hashed"
    for caveat in run["report"]["caveats"]:
        assert caveat, "caveats are how this repo refuses to overclaim"


def test_the_suggested_note_distinguishes_the_submission(run):
    note = run["report"]["submission"]["suggested_note"]
    assert "synthetic-union" in note
    assert "union" in note and "k=" in note, "the Note must say how the field was combined"
