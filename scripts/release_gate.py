#!/usr/bin/env python3
"""Fail-closed competition release gate for every published GeoTIFF.

This does not validate scientific claims or grant selector approval. It combines
already-recorded lane decisions and audits linked historical run cards so a
format PASS or historical run-card 'promote' field can never be mistaken for
permission to upload. The generated receipt is suitable for the Pages site.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANES = ROOT / "docs" / "data" / "lanes.json"
DEFAULT_OUT = ROOT / "docs" / "data" / "submission-authorization.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_run_card(run_card_rel: str | None, root: Path = ROOT) -> dict:
    """Audit a frozen historical run card without mutating it in place."""
    if not run_card_rel:
        return {
            "run_card": None,
            "present": False,
            "recorded_verdict": None,
            "recorded_safe_to_submit": None,
            "literal_duplicate_recorded": False,
            "real_model_forward_overlap_exceeds_070": False,
            "registry_incomplete": False,
            "holdout_invalid_for_promotion": False,
            "historical_promotion_conflict": False,
            "guardrail_clear": False,
        }
    rc_path = root / run_card_rel
    if not rc_path.is_file():
        return {
            "run_card": run_card_rel,
            "present": False,
            "recorded_verdict": None,
            "recorded_safe_to_submit": None,
            "literal_duplicate_recorded": False,
            "real_model_forward_overlap_exceeds_070": False,
            "registry_incomplete": True,
            "holdout_invalid_for_promotion": False,
            "historical_promotion_conflict": False,
            "guardrail_clear": False,
        }
    try:
        card = json.loads(rc_path.read_text())
    except (OSError, ValueError):
        return audit_run_card(None, root=root)
    if not isinstance(card, dict):
        return audit_run_card(None, root=root)
    verdict = card.get("verdict")
    safe_sub = bool(card.get("safe_to_download_and_submit") or card.get("ok_to_submit"))

    ua = card.get("uniqueness_audit") or {}
    rcmp = card.get("registry_comparison") or {}
    uvr = card.get("uniqueness_vs_registry") or {}
    cov = card.get("correlation_overlap_vs_registry") or {}
    reg = card.get("registry") or {}
    lit_read = rcmp.get("literal_reading") or {}

    literal_dup = bool(
        card.get("protocol_unique") is False
        or card.get("policy_i17_literal_verdict") == "negative"
        or ua.get("template_literal_reading_duplicate") is True
        or rcmp.get("template_literal_reading_duplicate") is True
        or lit_read.get("duplicate") is True
        or cov.get("duplicate_stop") is True
        or (cov.get("final_dots_max_forward_overlap_3px") or 0) > 0.70
        or (reg.get("surface_max_forward_overlap_3px") or 0) > 0.70
        or "fails closed" in str(uvr.get("unfiltered_dense_witness_note") or "").lower()
        or (reg.get("duplicate_count") or 0) > 0
        or (reg.get("duplicate_rasters_in_checked_subset") or 0) > 0
        or (card.get("literal_reading_firings_final_dots") or 0) > 0
    )
    real_fwd = uvr.get("post_placement_dots_max_forward_3px_overlap_real_models")
    real_fwd_exceeds = bool(real_fwd is not None and float(real_fwd) > 0.70)
    cov_phases = [v for v in cov.values() if isinstance(v, dict)]
    cov_incomplete = any(v.get("complete_accessible_scan") is False for v in cov_phases)
    cov_literal_dup = any(
        (v.get("dot_duplicate_count") or 0) > 0
        or (v.get("max_forward_overlap_3px") or 0) > 0.70
        for v in cov_phases
    )
    reg_incomplete = bool(
        reg.get("complete") is False
        or reg.get("full_registry_certified") is False
        or cov_incomplete
        or (rcmp.get("unreadable_registry_files") or 0) > 0
    )
    literal_dup = literal_dup or cov_literal_dup
    holdout_invalid = bool(card.get("holdout_valid_for_promotion") is False)

    guardrail_clear = not (
        literal_dup or real_fwd_exceeds or reg_incomplete or holdout_invalid
    )
    conflict = bool(
        (verdict == "promote" or safe_sub)
        and (not guardrail_clear or card.get("approved_for_weekly_slot") is not True)
    )
    return {
        "run_card": run_card_rel,
        "present": True,
        "recorded_verdict": verdict,
        "recorded_safe_to_submit": safe_sub,
        "literal_duplicate_recorded": literal_dup,
        "real_model_forward_overlap_3px": real_fwd,
        "real_model_forward_overlap_exceeds_070": real_fwd_exceeds,
        "registry_incomplete": reg_incomplete,
        "holdout_invalid_for_promotion": holdout_invalid,
        "historical_promotion_conflict": conflict,
        "guardrail_clear": guardrail_clear,
    }


def assess(row: dict, docs: Path, root: Path = ROOT) -> dict:
    path = docs / row["file"]
    rc_audit = audit_run_card(row.get("run_card"), root=root)
    checks = {
        "artifact_exists": path.is_file(),
        "run_card_present": rc_audit["present"],
        "format_valid": row.get("format_valid") is True,
        "literal_protocol_unique": (
            row.get("protocol_unique") is True
            and not rc_audit["literal_duplicate_recorded"]
            and not rc_audit["real_model_forward_overlap_exceeds_070"]
            and not rc_audit["registry_incomplete"]
        ),
        "beats_accepted_holdout_best": (
            row.get("beats_holdout_best") is True
            and not rc_audit["holdout_invalid_for_promotion"]
        ),
        "scientific_verdict_promote": (
            row.get("verdict") == "promote"
            and rc_audit["recorded_verdict"] == "promote"
            and rc_audit["guardrail_clear"]
        ),
        "selector_approved": row.get("selector_approved") is True,
    }
    expected = row.get("sha256")
    measured = sha256(path) if path.is_file() else None
    checks["sha256_matches"] = bool(expected and measured == expected)
    authorized = all(checks.values()) and rc_audit["guardrail_clear"]
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "lane": row["lane"],
        "file": row["file"],
        "sha256": measured,
        "checks": checks,
        "run_card_audit": rc_audit,
        "authorized_to_submit": authorized,
        "decision": "OK TO SUBMIT" if authorized else "NOT OK TO SUBMIT",
        "failed_gates": failed,
    }


def build(lanes_path: Path = DEFAULT_LANES, out: Path = DEFAULT_OUT) -> dict:
    source = json.loads(lanes_path.read_text())
    rows = [assess(row, ROOT / "docs", root=ROOT) for row in source["lanes"]]
    reconciliations = [
        {
            "lane": r["lane"],
            "run_card": r["run_card_audit"]["run_card"],
            "recorded_verdict": r["run_card_audit"]["recorded_verdict"],
            "recorded_safe_to_submit": r["run_card_audit"]["recorded_safe_to_submit"],
            "historical_promotion_conflict": r["run_card_audit"]["historical_promotion_conflict"],
            "operative_decision": r["decision"],
            "reconciliation_note": (
                "Frozen historical run card retains its original experiment fields for audit; "
                "superseded by fail-closed repository release gate."
                if r["run_card_audit"]["historical_promotion_conflict"]
                else "Historical run card and operative release gate both fail closed."
            ),
        }
        for r in rows
        if r["run_card_audit"]["run_card"] is not None
    ]
    receipt = {
        "schema": "gems-submission-authorization-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": (
            "Authorization requires artifact existence, matching SHA-256, local format PASS, "
            "literal protocol uniqueness (including <=0.70 real-model forward 3-px overlap and "
            "complete registry), improvement over the accepted holdout best, a promote "
            "verdict, and explicit selector approval. Missing fields or historical run-card "
            "uniqueness/completeness failures fail closed."
        ),
        "repository_authorized_file_count": sum(r["authorized_to_submit"] for r in rows),
        "repository_decision": "OK TO SUBMIT" if any(r["authorized_to_submit"] for r in rows) else "NOT OK TO SUBMIT",
        "historical_run_card_reconciliation": reconciliations,
        "artifacts": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=Path, default=DEFAULT_LANES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--verify-fail-closed",
        action="store_true",
        help="Exit 0 when the gate runs cleanly and confirms 0 files are falsely authorized.",
    )
    args = parser.parse_args()
    receipt = build(args.lanes, args.out)
    print(json.dumps({
        "decision": receipt["repository_decision"],
        "authorized_files": receipt["repository_authorized_file_count"],
        "receipt": str(args.out),
    }, indent=2))
    if args.verify_fail_closed:
        return 0 if receipt["repository_authorized_file_count"] == 0 else 1
    return 0 if receipt["repository_authorized_file_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
