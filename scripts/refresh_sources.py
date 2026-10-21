#!/usr/bin/env python3
"""Refresh a GitHub-owner source inventory without treating it as score evidence.

The project owner creates new ``GEMSDOE``-numbered repositories over time.  A
hard-coded endpoint list silently went stale at GEMSDOE78 even though the brief
listed later lanes.  This job discovers every *public* repository matching the
family naming convention first, then records a pinned page/tree inventory for
each.  It deliberately does not download rasters, validate scientific claims,
score a model, change a candidate, or spend a competition slot.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OWNER = "buffedlizard55-lab"
# Matches the historical spellings GEMSDOE, GEMSDOE2, 5GEMSDOE, 55GEMSDOE,
# and future GEMSDOE79-style repositories.  It intentionally excludes e.g.
# LEARNGEMSDOE so the provenance scope remains the submitted-family inventory.
FAMILY_REPOSITORY = re.compile(r"^(?:GEMSDOE\d*|\d+GEMSDOE)$", re.IGNORECASE)
DISCOVERY_ENDPOINT = f"users/{OWNER}/repos?per_page=100&type=owner"

OFFICIAL = [
    ("Organizer reference", "https://github.com/drivendataorg/gems-prize-reference-solution"),
    ("Competition problem", "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/"),
    ("Leaderboard", "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/"),
    ("Submission/data portal", "https://www.drivendata.org/competitions/306/competition-doe-gems/data/"),
    ("GeoDAWN official release", "https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and"),
    ("INGENIOUS GDR", "https://gdr.openei.org/submissions/1391"),
    ("Competition report", "https://docs.nlr.gov/docs/fy26osti/96647.pdf"),
    ("INGENIOUS project", "https://gbcge.org/current-projects/ingenious/"),
    ("CRS reference", "https://epsg.io/32611"),
]


def gh_json(endpoint: str) -> Any:
    process = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, timeout=40)
    if process.returncode:
        raise RuntimeError(process.stderr.strip()[:400])
    return json.loads(process.stdout)


def _natural_family_key(target: dict[str, str]) -> tuple[int, str]:
    """Order GEMSDOE, GEMSDOE2, 5GEMSDOE, ... by intended numeric lane number."""
    name = target["name"].upper()
    if name.startswith("GEMSDOE"):
        suffix = name.removeprefix("GEMSDOE")
    else:
        suffix = name.removesuffix("GEMSDOE")
    return (int(suffix) if suffix else 1, target["name"].casefold())


def discover_public_family_repositories() -> list[dict[str, str]]:
    """Discover all public sibling repositories across REST pagination.

    A discovery failure raises rather than publishing a falsely "up-to-date"
    partial inventory.  ``gh api --paginate`` emits one tab-delimited row per
    page/item, avoiding a second unpublished hard-coded repository list.
    """
    process = subprocess.run(
        [
            "gh", "api", "--paginate", DISCOVERY_ENDPOINT,
            "--jq", ".[] | select(.private == false) | [.name, .default_branch] | @tsv",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if process.returncode:
        raise RuntimeError(f"GitHub family discovery failed: {process.stderr.strip()[:400]}")
    targets: list[dict[str, str]] = []
    for line in process.stdout.splitlines():
        name, separator, branch = line.partition("\t")
        if not separator or not name or not branch:
            raise RuntimeError(f"GitHub family discovery emitted malformed row: {line!r}")
        if FAMILY_REPOSITORY.fullmatch(name):
            targets.append({"name": name, "default_branch": branch})
    if not targets:
        raise RuntimeError("GitHub family discovery found zero public GEMSDOE repositories")
    names = [target["name"] for target in targets]
    if len(names) != len(set(names)):
        raise RuntimeError("GitHub family discovery returned duplicate repository names")
    return sorted(targets, key=_natural_family_key)


def repo_health(target: dict[str, str]) -> dict[str, Any]:
    """Inventory a public owner mirror; this is never organizer score verification."""
    name = target["name"]
    branch = target["default_branch"]
    row: dict[str, Any] = {
        "repo": name,
        "default_branch": branch,
        "url": f"https://github.com/{OWNER}/{name}",
        "claim_class": "OWNER-MIRROR; not organizer score verification",
    }
    try:
        commit = gh_json(f"repos/{OWNER}/{name}/commits/{branch}")["sha"]
        tree = gh_json(f"repos/{OWNER}/{name}/git/trees/{commit}?recursive=1")
        row.update(status="retrieved", commit=commit, tree_sha=tree["sha"], truncated=tree.get("truncated", False))
        blobs = [entry for entry in tree["tree"] if entry["type"] == "blob"]
        page = next((entry for path in ("docs/index.html", "index.html", "README.md")
                     for entry in blobs if entry["path"] == path), None)
        row["review_page"] = f'{row["url"]}/blob/{commit}/{page["path"]}' if page else None
        row["review_page_blob_sha"] = page["sha"] if page else None
        row["tif_paths"] = [entry["path"] for entry in blobs
                            if entry["path"].lower().endswith(".tif") and "download" in entry["path"]]
        row["download_tif_count"] = len(row["tif_paths"])
        # Read actual page bytes for the owner-family sites; do not infer a
        # submission receipt merely because a page happens to contain a number.
        if page:
            page_process = subprocess.run(
                [
                    "gh", "api", f'repos/{OWNER}/{name}/git/blobs/{page["sha"]}',
                    "-H", "Accept: application/vnd.github.raw",
                ],
                capture_output=True,
                timeout=40,
            )
            if page_process.returncode:
                raise RuntimeError(page_process.stderr.decode(errors="replace")[:200])
            row["page_content_sha256"] = hashlib.sha256(page_process.stdout).hexdigest()
            text = page_process.stdout.decode("utf-8", errors="replace")
            terms = ["0.2778", "0.3195", "0.4413", "ORGANIZER-CONFIRMED", "HOLDOUT-DTI"]
            row["page_mentions"] = {term: (term in text) for term in terms}
        row["authenticated_score_receipt"] = False
    except Exception as exc:  # Preserve per-repository retrieval errors in the dated inventory.
        row.update(status="error", error=str(exc))
    return row


def official_health(pair: tuple[str, str]) -> dict[str, Any]:
    name, url = pair
    row: dict[str, Any] = {"name": name, "url": url, "score_verified": False}
    if url.startswith("https://github.com/"):
        try:
            data = gh_json("repos/drivendataorg/gems-prize-reference-solution/commits/HEAD")
            row.update(
                status="retrieved via GitHub API",
                commit=data["sha"],
                scope="Official reference code only; not portal, DTI-distance spec or leaderboard receipt",
            )
        except Exception as exc:
            row.update(status="error", error=str(exc))
    else:
        # Do not evade the sandbox allowlist or mislabel lack of access as source unavailability.
        row.update(
            status="not fetched by scheduled CI runner (restricted outbound hosts)",
            verification="Automated feed has no official-page check. Dated manual web-fetch review: docs/review-2026-10-21-h77l.md. Do not infer current scores from that snapshot.",
        )
    return row


def main() -> None:
    targets = discover_public_family_repositories()
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(repo_health, targets))
    report = {
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "GitHub owner-mirror/page freshness and artifact inventory, NOT scores, scientific validation, or full raster uniqueness",
        "repository_discovery": {
            "endpoint": f"https://api.github.com/{DISCOVERY_ENDPOINT}",
            "owner": OWNER,
            "public_only": True,
            "name_pattern": FAMILY_REPOSITORY.pattern,
            "matched_repository_count": len(targets),
            "matched_repositories": [target["name"] for target in targets],
            "failure_policy": "Fail the refresh instead of publishing a partial list when family discovery fails.",
        },
        "repos": rows,
        "official_sources": [official_health(pair) for pair in OFFICIAL],
        "score_policy": "No organizer-confirmed receipt is available in this inventory. Never treat owner webpage mentions as verified scores.",
        "weekly_cap": None,
        "weekly_cap_status": "CI does not re-check official rules PDF. Dated manual review 2026-10-21 observed three feedback submissions/week; see docs/review-2026-10-21-h77l.md. Zero slots consumed by this pipeline.",
    }
    for path in (ROOT / "evidence" / "source_health.json", ROOT / "docs" / "source-health.json"):
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Discovered {len(targets)} public family repositories; retrieved "
          f"{sum(row['status'] == 'retrieved' for row in rows)} page inventories")


if __name__ == "__main__":
    main()
