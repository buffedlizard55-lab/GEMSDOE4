"""scripts/fetch_bridge_parts.py — the runner-side data-placement gap, and its guard.

GEMSDOE4 commits `data/bridge/manifest.json` (and the two small rasters) but not the ~418 MB of
`*.part-*` files, because this platform caps a turn's patchset at ~128 MB.  The workflows test for
the *manifest* to decide whether the bridge is "in the tree", so on this repository they took the
local-assembly branch and failed at `assemble_data_bridge.py` — measured: `make-submission.yml` run
36186011727 failed at step 7 and skipped steps 8-14, committing an evidence record of nulls.

`fetch_bridge_parts.py` closes that gap.  What these tests pin is the part that matters for
correctness: **nothing is trusted on the strength of the download.**  A part whose sha256 does not
match the manifest pin is a hard failure, the mirror ref must be explicit (a branch name would make
a pinned artefact depend on a moving target), and `--check` is the no-network probe the workflows
can use to distinguish "bridge absent" from "bridge incomplete".
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_bridge_parts.py"


def _mod():
    spec = importlib.util.spec_from_file_location("fetch_bridge_parts", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bridge(tmp_path: Path, with_parts: bool = False, corrupt: bool = False) -> Path:
    """A miniature bridge: one file split into two parts, with the manifest's own sha256 pins."""
    b = tmp_path / "bridge"
    b.mkdir()
    whole = b"GEMS-DOE4-SYNTHETIC-BRIDGE-BYTES" * 64
    parts = [whole[: len(whole) // 2], whole[len(whole) // 2:]]
    names = ["synthetic.tif.part-000", "synthetic.tif.part-001"]
    manifest = {
        "generated_utc": "2026-09-25T00:00:00Z",
        "part_bytes": len(parts[0]),
        "files": [dict(name="synthetic.tif", canonical="synthetic.tif", bytes=len(whole),
                       sha256=hashlib.sha256(whole).hexdigest(),
                       parts=[dict(name=n, bytes=len(p), sha256=hashlib.sha256(p).hexdigest())
                              for n, p in zip(names, parts)])],
    }
    (b / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if with_parts:
        for n, p in zip(names, parts):
            data = p[::-1] if (corrupt and n == names[0]) else p
            (b / n).write_bytes(data)
    return b


# --------------------------------------------------------------------------- the no-network probe
def test_check_fails_when_the_parts_are_absent(tmp_path, capsys):
    """The exact state of this repository, and the reason the workflows broke."""
    m = _mod()
    b = _bridge(tmp_path, with_parts=False)
    rc = m.main(["--bridge", str(b), "--check"])
    assert rc == 1, "an incomplete bridge must fail, not quietly pass"
    err = capsys.readouterr().err
    assert "2 missing" in err


def test_check_passes_when_every_part_matches_its_pin(tmp_path, capsys):
    m = _mod()
    b = _bridge(tmp_path, with_parts=True)
    assert m.main(["--bridge", str(b), "--check"]) == 0
    out = capsys.readouterr().out
    assert "all 2 parts verified" in out


def test_check_fails_when_a_part_does_not_match_its_pin(tmp_path, capsys):
    m = _mod()
    b = _bridge(tmp_path, with_parts=True, corrupt=True)
    assert m.main(["--bridge", str(b), "--check"]) == 1
    err = capsys.readouterr().err
    assert "MISMATCH" in err and "1 mismatched" in err


# --------------------------------------------------------------------------------- the mirror ref
class _Args:
    """Stand-in for the parsed CLI flags, so `resolve_mirror_ref` can be tested without a fetch."""

    mirror_repo = None
    mirror_ref = None


def test_the_mirror_ref_must_be_explicit(tmp_path):
    """A sha256-pinned artefact must not depend on a branch name that moves."""
    m = _mod()
    b = _bridge(tmp_path, with_parts=False)
    with pytest.raises(SystemExit) as exc:
        m.main(["--bridge", str(b)])
    assert "no mirror ref given" in str(exc.value)


def test_the_recorded_mirror_is_actually_read_from_the_bridge_directory(tmp_path):
    """The bug that made the first fix fail on the runner, measured in run 36187101169.

    `MIRROR_RECORD` was resolved as `bridge / MIRROR_RECORD.split("/", 1)[1]`, i.e.
    `data/bridge/bridge/mirror.json` - a path that can never exist.  The record was therefore
    always invisible, so every run refused with "no mirror ref given" and step 7 failed exactly as
    it had before the fix.  The record is looked up *relative to the bridge directory*, and this
    test proves the lookup resolves to the committed file rather than to a doubled path.
    """
    m = _mod()
    b = ROOT / "data/bridge"
    repo, ref = m.resolve_mirror_ref(b, _Args())
    rec = json.loads((b / "mirror.json").read_text())
    assert (repo, ref) == (rec["repo"], rec["ref"]), (
        "the mirror record must be read from <bridge>/mirror.json, not from a doubled path")
    assert ref == "cceebbdcf9a7d2890bb0665defcb54dfc66ae452"


def test_the_recorded_mirror_is_pinned_to_a_commit_sha():
    """The repo's own mirror record: a 40-hex ref, not 'main'."""
    m = _mod()
    rec = json.loads((ROOT / "data/bridge/mirror.json").read_text())
    assert set(rec) >= {"repo", "ref"}
    assert len(rec["ref"]) == 40 and all(c in "0123456789abcdef" for c in rec["ref"]), \
        "the mirror ref must be a commit sha, or a pinned artefact depends on a moving target"
    assert rec["repo"] == "buffedlizard55-lab/GEMSDOE"
    # the URL the fetcher would actually build must point at that repo, that commit, that part
    url = m.RAW.format(repo=rec["repo"], ref=rec["ref"],
                       part="gems-geodawn-numerical-features.tif.part-000")
    assert url == ("https://raw.githubusercontent.com/buffedlizard55-lab/GEMSDOE/"
                   "cceebbdcf9a7d2890bb0665defcb54dfc66ae452/"
                   "data/bridge/gems-geodawn-numerical-features.tif.part-000")


# --------------------------------------------------------------------------- the real repository
def test_the_committed_bridge_really_is_incomplete_here():
    """Documenting the state this script exists for, so the gap cannot be forgotten.

    The property that matters lives in git's INDEX, not on this checkout's disk: a checkout may
    legitimately have restored the parts locally (`fetch_bridge_parts.py` does exactly that, and
    this repository's own dev sandbox now runs with them in place), and the fetcher must stay in
    the workflows regardless.  What must never happen is the ~418 MB travelling by commit - that
    is what breaks the platform's ~128 MB patchset cap.
    """
    b = ROOT / "data/bridge"
    manifest = json.loads((b / "manifest.json").read_text())
    parts = [p for f in manifest["files"] for p in f.get("parts", [])]
    assert parts, "the manifest must list parts"
    tracked = set(subprocess.run(["git", "ls-files", "--", "data/bridge"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.split())
    tracked_parts = [p for p in parts if f"data/bridge/{p['name']}" in tracked]
    assert len(tracked_parts) < len(parts), (
        "every part is now committed - re-check whether the workflows still need "
        "fetch_bridge_parts.py, and update this test's docstring")


def test_the_workflows_call_the_fetcher_before_the_assembler():
    """The guard is the fix; a workflow that forgets it is the bug that just happened.

    Only workflows that actually *run* the assembler count.  `place-competition-data.yml` is the
    generating side - it downloads the rasters from the data-tab Dropbox mirrors and builds the
    bridge - so its only mention of the assembler is a documentation comment, and requiring a
    fetcher there would be noise.
    """
    ASM = "python scripts/assemble_data_bridge.py"
    FETCH = "python scripts/fetch_bridge_parts.py"
    for f in sorted((ROOT / ".github/workflows").glob("*.yml")):
        lines = [ln for ln in f.read_text().split("\n")
                 if ln.strip() and not ln.strip().startswith("#")]
        if not any(ln.strip() == ASM for ln in lines):
            continue
        text = "\n".join(lines)
        assert FETCH in text, f"{f.name} assembles the bridge without restoring its parts"
        assert text.find(FETCH) < text.find(ASM), \
            f"{f.name} must fetch the parts before assembling them"
