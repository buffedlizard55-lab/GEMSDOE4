"""Tests for scripts/check_pages_urls.py — "which URL form is live?" must be measured, not assumed.

WHY THIS FILE EXISTS.  This repository publishes through two GitHub Pages mechanisms at once (a
legacy build from `main:/` and the Actions upload of `docs/`), and measured 2026-09-26 they answered
differently: one URL form served a pre-merge copy of a page and intermittently 404'd while the other
served the current build.  The probe that detects this runs on a runner (the sandbox cannot reach
`github.io`), so its *analysis* is what the tests can and must pin:

* a 200 is not "live" unless the page's own `<title>` comes back - otherwise a redirect to the 404
  page or to another site would read as success, which is the exact failure mode this repo keeps
  finding in its own tooling;
* `live_form` must be `None` when neither form is fully live, and a disagreement between the two
  forms must be reported as such rather than averaged away;
* a partially-live form is reported with the pages that are missing, so a reader can follow the link
  that works.

The fixtures below are recorded probes, not live fetches: the analysis must be testable without the
network, and a test that needed github.io would be the flakiest test in the suite.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "check_pages_urls.py"


def _mod():
    spec = importlib.util.spec_from_file_location("check_pages_urls", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PAGES = ("index.html", "results.html")


def _records(prefixed, root):
    """`[(status, title), ...]` per form, mirroring what a real probe records."""
    out = []
    for form, spec in (("prefixed", prefixed), ("root", root)):
        for page, (status, title) in zip(PAGES, spec):
            out.append(dict(form=form, page=page, url=f"u://{form}/{page}",
                            status=status, bytes=100 if status == 200 else None,
                            title=title, final_url=f"u://{form}/{page}"))
    return out


TITLES = {"index.html": "Index — GEMS Prize", "results.html": "Results — GEMS Prize"}


def _served(title: str, status: int = 200):
    """A page that answers with its own title (i.e. genuinely served)."""
    return (status, title)


def _gone(status: int = 404):
    """A page that does not answer with the real page - a status the probe records as-is."""
    return (status, None)


def test_the_root_form_live_and_the_prefixed_form_stale_is_reported_as_a_disagreement():
    """The measured 2026-09-26 shape: root form current, /docs/ form serving an older build."""
    m = _mod()
    stale = (_served(TITLES["index.html"]),
             _served("Results — GEMS Prize (2026-09-20 build)"))
    current = (_served(TITLES["index.html"]), _served(TITLES["results.html"]))
    v = m.analyse(_records(stale, current), TITLES)
    assert v["live_forms"] == ["root"]
    assert v["live_form"] == "root"
    assert v["agree"] is False, "two different builds is the irregularity - it must be stated"
    assert v["per_form"]["prefixed"]["live"] == 1, "the one page that matches still counts as live"
    assert "results.html" in v["per_form"]["prefixed"]["missing"]


def test_the_prefixed_form_live_and_the_root_form_absent_is_also_reported():
    """The legacy-only configuration: /docs/ works, the root form 404s.  Neither is 'wrong'."""
    m = _mod()
    current = (_served(TITLES["index.html"]), _served(TITLES["results.html"]))
    notfound = (_gone(), _gone())
    v = m.analyse(_records(current, notfound), TITLES)
    assert v["live_form"] == "prefixed"
    assert v["agree"] is False
    assert v["per_form"]["root"]["live"] == 0
    assert sorted(v["per_form"]["root"]["missing"]) == list(PAGES)


def test_both_forms_live_is_agreement_and_neither_live_is_not_dressed_up():
    m = _mod()
    current = (_served(TITLES["index.html"]), _served(TITLES["results.html"]))
    both = m.analyse(_records(current, current), TITLES)
    assert both["live_form"] == "both", "when both forms serve the same build, say so"
    assert both["agree"] is True
    assert sorted(both["live_forms"]) == ["prefixed", "root"]
    dead = m.analyse(_records((_gone(None), _gone(None)), (_gone(None), _gone(None))), TITLES)
    assert dead["live_form"] is None and dead["live_forms"] == []
    assert dead["per_form"]["prefixed"]["live"] == 0


def test_a_200_without_the_page_title_is_not_live():
    """A redirect to a holding page, or a soft 404, must not be counted as the page being served."""
    m = _mod()
    soft = (_served("Index — GEMS Prize", 200)[0:1] + (200,))  # status 200, wrong title
    soft = ((200, "Some other site"), (200, "Some other site"))
    v = m.analyse(_records(soft, soft), {"index.html": "Something else", "results.html": None})
    assert v["live_form"] is None
    assert v["per_form"]["prefixed"]["live"] == 0


def test_the_url_forms_are_the_two_real_ones():
    m = _mod()
    assert m.url_for("https://x/GEMSDOE4", "prefixed", "index.html") == \
        "https://x/GEMSDOE4/docs/index.html"
    assert m.url_for("https://x/GEMSDOE4", "root", "index.html") == "https://x/GEMSDOE4/index.html"
    assert m.FORMS == ("prefixed", "root")


def test_every_published_page_is_covered_by_the_probe():
    """The probed set is read from docs/, so a new page cannot be published unprobed."""
    m = _mod()
    names = m.pages(ROOT)
    assert "index.html" in names and "results.html" in names and len(names) >= 10
    assert all(n.endswith(".html") for n in names)


def test_the_written_evidence_names_the_human_action():
    """The consequence is a repository SETTING; the file has to say so or the finding is inert."""
    m = _mod()
    out = ROOT / "data/evidence/pages_urls.json"
    if not out.exists():
        pytest.skip("no runner probe recorded yet - the workflow writes this on push to main")
    d = json.loads(out.read_text())
    assert d["question"] and d["covers"]
    assert "Settings" in d["human_action"] and "Pages" in d["human_action"]
    assert set(d["per_form"]) == {"prefixed", "root"}
