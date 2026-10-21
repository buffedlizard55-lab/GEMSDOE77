#!/usr/bin/env python3
"""Apply this session's discriminating-criterion audits of the incumbent files (H77D, H77-Step) to docs/data/lanes.json.

Reads evidence/<tag>_uniqueness.json (written by scripts/h79_uniqueness.py --merge --tag <tag> --build-file -)
and sets, for the matching lane entry: protocol_unique_like_for_like, uniqueness_measured_by, status, upload.
The status is the same mechanical three-state rule as everywhere else (docs/policy_uniqueness.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_home as BH  # noqa: E402

TAGS = {"h77d": "h77d.html", "h77step": "h77-step.html"}


def main() -> None:
    lanes_path = ROOT / "docs" / "data" / "lanes.json"
    lanes = json.loads(lanes_path.read_text())
    for tag, page in TAGS.items():
        f = ROOT / "evidence" / f"{tag}_uniqueness.json"
        if not f.exists():
            print(f"{tag}: no audit file, lane untouched")
            continue
        u = json.loads(f.read_text())
        lane = next(x for x in lanes["lanes"] if x["page"] == page)
        import hashlib
        file_sha = hashlib.sha256((ROOT / "docs" / lane["file"]).read_bytes()).hexdigest()
        lane.setdefault("sha256", file_sha)
        if u["sha256"] != file_sha or lane["sha256"] != file_sha:
            raise SystemExit(f"{tag}: audited sha {u['sha256'][:12]} / lane sha {lane['sha256'][:12]} != file sha {file_sha[:12]}")
        complete = u["phase2_final_dots"]["rasters_checked"] >= 899
        lfl = bool(u["protocol_unique_like_for_like"] and complete)
        worst = u["worst_discriminating_rows"][0]
        lane["protocol_unique_like_for_like"] = lfl
        lane["protocol_unique"] = bool(u["protocol_unique_literal"])
        lane["uniqueness_measured_by"] = (f"this session (2026-10-21): gems57.uniqueness final dots vs {u['phase2_final_dots']['rasters_checked']} cached rasters, "
                                          f"discriminating criterion (registry_halo_coverage.json); max rho {u['phase2_final_dots']['max_rho']:.3f}; "
                                          f"max 3 px forward overlap vs discriminating rasters {u['max_forward_overlap_discriminating']:.3f} "
                                          f"({worst['repo']} {Path(worst['submission']).name}, halo coverage {worst['halo_coverage_allowed']:.3f}); evidence/{tag}_uniqueness.json")
        lane.pop("status", None)
        st = BH.status_of(lane)
        lane["status"] = st
        if st == "CLEARED_FOR_SELECTOR":
            lane["upload"] = ("OK to submit if the selector picks it: format PASS, unique under the published rule (re-measured this session under the "
                              f"discriminating criterion: max overlap {u['max_forward_overlap_discriminating']:.3f} <= 0.70, rho {u['phase2_final_dots']['max_rho']:.3f} <= 0.90), "
                              "beats the fold-fair controls per the lane's own card. Slot decision = selector.")
        else:
            lane["upload"] = ("Not OK to submit: " + ("uniqueness FAILS under the discriminating criterion re-measured this session "
                                                      f"(max 3 px forward overlap {u['max_forward_overlap_discriminating']:.3f} > 0.70 vs {worst['repo']} "
                                                      f"{Path(worst['submission']).name}, halo coverage {worst['halo_coverage_allowed']:.3f})"
                                                      if not lfl else "a gate other than uniqueness is missing") + ". Download for inspection only.")
        print(f"{tag}: lfl={lfl} status={st} max_fwd_disc={u['max_forward_overlap_discriminating']:.3f} rho={u['phase2_final_dots']['max_rho']:.3f}")
    lanes_path.write_text(json.dumps(lanes, indent=1))


if __name__ == "__main__":
    main()
