#!/usr/bin/env python3
"""Which URL actually serves this site right now — `/docs/<page>` or `/<page>`?

WHY THIS EXISTS
---------------
GitHub Pages for this repository has TWO publish mechanisms configured at once, and that is a real
(flagged) irregularity rather than a hypothetical:

  * the **legacy** Jekyll builder, whose source is `main:/` — it publishes the repository root, so
    every page lives at `.../GEMSDOE4/docs/<page>.html`;
  * `.github/workflows/pages.yml`, which uploads `docs/` as a Pages artefact — it publishes the
    CONTENTS of `docs/`, so the same page lives at `.../GEMSDOE4/<page>.html`.

`GET /repos/…/pages` reports `build_type: legacy, source: main:/` *and* the Actions run reports
success, and in practice the two forms answer at different times: measured 2026-09-26, the
docs-prefixed URL served a PRE-MERGE copy of `results.html` (and intermittently 404'd) while the
un-prefixed URL served the current build.  A reader who follows a stale link therefore sees either an
old page or a 404, and no amount of care inside the repository fixes that — the fix is one setting in
the repository's Pages configuration, which only a human with admin rights can change.

So this script MEASURES which form is live, for every published page, and records the answer as
committed evidence.  It cannot run in the dev sandbox (`github.io` is outside the egress allowlist),
so it runs on a GitHub runner — the same pattern `scripts/verify_links.py` uses for the data catalog,
for the same reason.

WHAT IT REPORTS, AND WHAT IT REFUSES TO REPORT
----------------------------------------------
* Every page is fetched from BOTH forms in one run; a form that answers 200 with the page's own
  `<title>` is live, anything else is recorded with its status code.
* A 200 is not enough on its own: without the title check, a redirect to the 404 page or to another
  site would look like success.
* The script never guesses which mechanism "should" win. It states which one answered, and whether
  the two forms agree — a disagreement is the irregularity, and hiding it would leave the next
  session rediscovering it.

Usage:
    python scripts/check_pages_urls.py                       # probe both forms of every page
    python scripts/check_pages_urls.py --base https://buffedlizard55-lab.github.io/GEMSDOE4
    python scripts/check_pages_urls.py --offline-file <json>  # analyse a recorded probe (tests)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE = "https://buffedlizard55-lab.github.io/GEMSDOE4"
UA = ("Mozilla/5.0 (compatible; gems-prize-pages-probe/1.0; "
      "+https://github.com/buffedlizard55-lab/GEMSDOE4)")
# The two forms under test, in the order a human is most likely to try them.  `prefixed` is the
# legacy/Jekyll form (site root = repository root), `root` is the Actions-artefact form (site root =
# docs/).  They are not interchangeable, and the whole point of this script is that fact.
FORMS = ("prefixed", "root")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def pages(root: Path) -> list[str]:
    """Every published page, as a path relative to `docs/`."""
    return sorted(p.name for p in (root / "docs").glob("*.html"))


def url_for(base: str, form: str, page: str) -> str:
    return f"{base}/docs/{page}" if form == "prefixed" else f"{base}/{page}"


def probe(url: str, timeout: int = 30) -> dict:
    """One GET.  A page is 'live' only if it answers 200 AND carries a <title>."""
    if url.startswith("file://"):
        # offline mode for tests: the "URL" is a path under the fixture root
        body = Path(url[7:]).read_bytes()
        return dict(status=200, bytes=len(body), title=_title(body.decode("utf-8", "replace")),
                    final_url=url)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return dict(status=int(r.status), bytes=len(body),
                        title=_title(body.decode("utf-8", "replace")),
                        final_url=r.geturl())
    except urllib.error.HTTPError as e:
        return dict(status=int(e.code), bytes=None, title=None, final_url=e.url)
    except Exception as e:                                   # noqa: BLE001 - reported, not raised
        return dict(status=None, bytes=None, title=None, final_url=url,
                    error=f"{type(e).__name__}: {e}")


def _title(html: str) -> str | None:
    m = TITLE_RE.search(html)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def analyse(records: list[dict], expected_titles: dict[str, str | None]) -> dict:
    """Turn raw probes into the two statements a reader needs, without dressing them up.

    * `live_form`: the form whose every page answered 200 with the page's own title.  `None` when
      neither form did - which is a finding, not a failure to report.
    * `agree`: whether both forms serve the same page titles (i.e. whether the irregularity is
      currently *visible* to a reader, as opposed to merely configured).
    """
    per_form: dict[str, dict] = {}
    for form in FORMS:
        rows = [r for r in records if r["form"] == form]
        live = [r for r in rows
                if r["status"] == 200 and r["title"] and expected_titles.get(r["page"])
                and r["title"] == expected_titles[r["page"]]]
        per_form[form] = dict(probed=len(rows), live=len(live),
                              live_pages=sorted(r["page"] for r in live),
                              missing=sorted(r["page"] for r in rows if r not in live))
    live_forms = [f for f, v in per_form.items() if v["probed"] and v["live"] == v["probed"]]
    # The forms agree when they serve the SAME set of pages - which is true both when both are fully
    # live and when neither is.  A set that differs between the two forms (one stale, one current) is
    # the irregularity this script exists to catch.
    agree = ({tuple(per_form[f]["live_pages"]) for f in FORMS}).__len__() == 1
    return dict(per_form=per_form,
                # "which form do I quote?": a single fully-live form, `both` when both are, and
                # None when neither is - never a silent pick between two different builds.
                live_form=(live_forms[0] if len(live_forms) == 1
                           else ("both" if len(live_forms) == len(FORMS) else None)),
                live_forms=live_forms, agree=bool(agree))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help="site root (no trailing /docs)")
    ap.add_argument("--root", default=str(ROOT), help="checkout holding docs/")
    ap.add_argument("--out", default="data/evidence/pages_urls.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--offline-file", default=None,
                    help="analyse a previously recorded probe file instead of fetching (tests)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    names = pages(root)
    if not names:
        print(f"no pages under {root / 'docs'}")
        return 1

    if a.offline_file:
        records = json.loads(Path(a.offline_file).read_text())["records"]
    else:
        jobs = [(form, page) for form in FORMS for page in names]
        with ThreadPoolExecutor(max_workers=int(a.workers)) as pool:
            records = list(pool.map(
                lambda fp: dict(form=fp[0], page=fp[1], url=url_for(a.base, fp[0], fp[1]),
                                **probe(url_for(a.base, fp[0], fp[1]), timeout=int(a.timeout))),
                jobs))

    expected = {}
    for page in names:
        expected[page] = _title((root / "docs" / page).read_text(errors="replace"))
    verdict = analyse(records, expected)

    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="scripts/check_pages_urls.py",
        base=a.base,
        question="which URL form actually serves this site: .../docs/<page> or .../<page>?",
        why=("GitHub Pages has both a legacy build (source main:/ -> pages live under /docs/) and the "
             "Actions upload of docs/ (-> pages live at the root) configured at once; measured "
             "2026-09-26 the two answered differently and the /docs/ form also 404'd intermittently"),
        forms={form: f"{a.base}/docs/<page>" if form == "prefixed" else f"{a.base}/<page>"
               for form in FORMS},
        covers=names,
        **verdict,
        records=records,
        consequence=(
            "A stale or 404 link to a page that says 'upload this file' is worse than no link. The "
            "fix is a repository SETTING (Pages -> Build and deployment -> Source): picking "
            "'GitHub Actions' makes the root form canonical and retires the legacy build; picking "
            "'Deploy from a branch' with /docs makes the prefixed form canonical. Either is fine - "
            "having both is not. This script cannot change it; it can only prove, every run, which "
            "one a reader is getting."),
        human_action=("GitHub -> repo Settings -> Pages -> Build and deployment -> Source. Until that "
                      "is set, quote the form this file reports as live, and treat the other as "
                      "unverified."),
    )
    dest = root / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str) + "\n")

    for form, v in verdict["per_form"].items():
        print(f"[pages-urls] {form:<9} live {v['live']}/{v['probed']}"
              + (f"  missing: {', '.join(v['missing'][:4])}"
                 + (" ..." if len(v["missing"]) > 4 else "") if v["missing"] else ""))
    print(f"[pages-urls] live form: {verdict['live_form'] or 'NEITHER (see records)'}"
          f"   agree: {verdict['agree']}")
    print(f"[pages-urls] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
