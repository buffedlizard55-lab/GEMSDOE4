#!/usr/bin/env python3
"""Restore the pinned data-bridge PARTS this repository does not commit, from a mirror repo.

WHY THIS EXISTS
---------------
The official competition rasters travel as sha256-pinned git "parts" under `data/bridge/`
(GitHub rejects blobs >= 100 MB, and the feature stack is 418,912,844 B).  Every workflow that
touches the rasters runs `scripts/assemble_data_bridge.py`, which needs those parts on disk.

GEMSDOE4 was created by copying the GEMSDOE tree *without* the ~418 MB of parts, because this
platform caps a turn's patchset around 128 MB.  The consequence, measured rather than assumed:
`make-submission.yml` run 36186011727 failed at step 7 ("Place the official rasters from the
pinned git bridge"), so steps 8-14 - re-validating the artifact, the CPU route, the payload
check, the site rebuild, the packaging - were all SKIPPED, and the run committed an evidence
record whose numbers are all `null`.  A missing input must fail loudly like that; it must not
silently ship a stale artifact.

This script closes the gap without committing 418 MB: it fetches each missing part from a mirror
repository at a **pinned ref** and verifies it against the sha256 the manifest already records.
Nothing is trusted on the strength of the download - a part whose hash does not match is a hard
failure, exactly as a corrupted part would be.

    python scripts/fetch_bridge_parts.py                 # fetch what is missing, verify all pins
    python scripts/fetch_bridge_parts.py --check         # verify only, no network
    python scripts/fetch_bridge_parts.py --mirror-repo OWNER/REPO --mirror-ref SHA

The mirror defaults to the repository the bridge was generated from
(`buffedlizard55-lab/GEMSDOE`), and the ref must be given explicitly or recorded in
`data/bridge/mirror.json` - a moving default ref is how a pinned artefact silently becomes
unpinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIRROR_REPO = "buffedlizard55-lab/GEMSDOE"
MIRROR_RECORD = "data/bridge/mirror.json"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/data/bridge/{part}"
CHUNK = 1 << 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(bridge: Path) -> dict:
    m = bridge / "manifest.json"
    if not m.exists():
        raise SystemExit(f"no manifest at {m} - is this the right bridge directory?")
    return json.loads(m.read_text())


def mirror_ref(bridge: Path, args) -> tuple[str, str]:
    """(repo, ref) for the mirror, with the ref required to be explicit and recorded."""
    rec = bridge / MIRROR_RECORD.split("/", 1)[1]
    recorded = {}
    if rec.exists():
        try:
            recorded = json.loads(rec.read_text())
        except Exception:                                       # pragma: no cover
            recorded = {}
    repo = args.mirror_repo or recorded.get("repo") or DEFAULT_MIRROR_REPO
    ref = args.mirror_ref or recorded.get("ref")
    if not ref:
        raise SystemExit(
            "no mirror ref given. Pass --mirror-ref <commit-sha> (or record it in "
            f"{rec}); a branch name would make a sha256-pinned artefact depend on a moving target.")
    return repo, str(ref)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bridge", default=str(ROOT / "data/bridge"))
    ap.add_argument("--mirror-repo", default=None)
    ap.add_argument("--mirror-ref", default=None)
    ap.add_argument("--check", action="store_true", help="verify the local parts only, no network")
    args = ap.parse_args(argv)
    bridge = Path(args.bridge)
    manifest = load_manifest(bridge)

    wanted = []          # (part dict, parent file dict)
    for f in manifest.get("files", []):
        for part in f.get("parts", []):
            wanted.append((part, f))
    if not wanted:
        print("manifest lists no parts - nothing to fetch")
        return 0

    repo, ref = ("", "") if args.check else mirror_ref(bridge, args)
    missing, wrong, ok = [], [], []
    for part, parent in wanted:
        p = bridge / part["name"]
        if not p.exists():
            missing.append((part, parent))
        elif p.stat().st_size != int(part["bytes"]) or sha256_file(p) != part["sha256"]:
            wrong.append((part, parent))
        else:
            ok.append(part["name"])

    for name in ok:
        print(f"[bridge] OK      {name}")
    for part, parent in wrong:
        print(f"[bridge] MISMATCH {part['name']} (size/hash differ from the manifest pin)",
              file=sys.stderr)

    if args.check:
        if missing or wrong:
            print(f"[bridge] --check: {len(missing)} missing, {len(wrong)} mismatched, "
                  f"{len(ok)} verified", file=sys.stderr)
            return 1
        print(f"[bridge] --check: all {len(ok)} parts verified against the manifest")
        return 0

    for part, parent in missing:
        url = RAW.format(repo=repo, ref=ref, part=part["name"])
        dest = bridge / part["name"]
        tmp = dest.with_suffix(dest.suffix + ".downloading")
        print(f"[bridge] fetch   {part['name']} <- {url}")
        try:
            with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as out:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            print(f"[bridge] FAILED  {part['name']}: {exc}", file=sys.stderr)
            return 1
        got = sha256_file(tmp)
        if got != part["sha256"]:
            tmp.unlink(missing_ok=True)
            print(f"[bridge] FAILED  {part['name']}: sha256 {got} != pin {part['sha256']}",
                  file=sys.stderr)
            return 1
        tmp.replace(dest)
        print(f"[bridge] verified {part['name']} ({int(part['bytes']):,} B, sha256 {got[:16]}...)")

    # the whole-file pins are the ones that matter: a part can be right and the assembly wrong
    for f in manifest.get("files", []):
        whole = bridge / f["name"]
        if whole.exists() and sha256_file(whole) != f["sha256"]:
            print(f"[bridge] FAILED  {f['name']}: assembled bytes do not match the pin",
                  file=sys.stderr)
            return 1
    print(f"[bridge] {len(ok)} already present, {len(missing)} fetched and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
