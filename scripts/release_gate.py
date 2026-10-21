#!/usr/bin/env python3
"""Fail-closed competition release gate for every published GeoTIFF.

This is deliberately stricter than a format checker.  It re-reads the exact
GeoTIFF with the pinned shared validator and combines that receipt with a
matching frozen run card, the literal registry gate, a promotion-valid holdout,
and an explicit selector decision.  A manifest assertion, a historical
``promote`` field, or a local format PASS can never independently authorize an
upload.

The gate does not validate scientific claims or grant selector approval.  Its
output is the machine-readable source for the Pages site's unambiguous
"OK / NOT OK TO SUBMIT" status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from gems57.grid import load_grid  # noqa: E402
from gems57.validate import validate  # noqa: E402

DEFAULT_LANES = ROOT / "docs" / "data" / "lanes.json"
DEFAULT_OUT = ROOT / "docs" / "data" / "submission-authorization.json"
METADATA_NOTE_LIMIT = 140
RHO_LIMIT = 0.90
OVERLAP_LIMIT = 0.70


def sha256(path: Path) -> str:
    """Return the SHA-256 of one artifact without loading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _at(card: dict[str, Any], *path: str) -> Any:
    value: Any = card
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _first_text(card: dict[str, Any], paths: Iterable[tuple[str, ...]]) -> str | None:
    for path in paths:
        value = _at(card, *path)
        if isinstance(value, str):
            return value
    return None


def card_submission_metadata(card: dict[str, Any]) -> dict[str, str | None]:
    """Normalize historical run-card metadata without mutating frozen cards.

    Older cards use a few documented field spellings.  A future card must use
    one of these explicit fields; guessing from a TIFF filename is intentionally
    not accepted by the release gate.
    """
    return {
        "name": _first_text(
            card,
            (
                ("submission_name",),
                ("submission_artifacts", "submission_name"),
                ("submission", "submission_name"),
                ("submission", "name"),
            ),
        ),
        "note": _first_text(
            card,
            (
                ("submission_note",),
                ("submission_note_le140",),
                ("submission_note_le_140",),
                ("note",),
                ("submission_artifacts", "submission_note"),
                ("submission_artifacts", "submission_note_le140"),
                ("submission_artifacts", "submission_note_le_140"),
                ("submission", "submission_note"),
                ("submission", "note"),
            ),
        ),
    }


def _walk_json_fields(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    """Yield every JSON field/value pair, retaining nested evidence schemas."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json_fields(child, path + (str(key),))
    elif isinstance(value, list):
        # Preserve the aggregate list as evidence (for a two-sided CI), then
        # recurse so nested dictionaries/lists remain discoverable as well.
        yield path, value
        for index, child in enumerate(value):
            yield from _walk_json_fields(child, path + (str(index),))
    else:
        yield path, value


def _promotion_holdout_evidence_complete(card: dict[str, Any]) -> bool:
    """Require the metadata that makes a promotion HOLDOUT-DTI auditable.

    A bare ``holdout_valid_for_promotion: true`` must not turn an undocumented
    scalar into a release decision.  Cards may have evolved nested arm schemas,
    so inspect the frozen ``holdout_dti`` object recursively while requiring the
    invariant evidence elements: HOLDOUT-DTI label, evaluator version, a
    positive withheld-positive count, a finite score, and a finite two-sided CI.
    """
    holdout = _as_dict(card.get("holdout_dti"))
    if not holdout:
        return False
    fields = list(_walk_json_fields(holdout))
    labels = [
        value for path, value in fields
        if path and path[-1].lower() in {"evidence_label", "evidence_class"}
    ]
    has_label = any(isinstance(value, str) and value.upper() == "HOLDOUT-DTI" for value in labels)
    has_evaluator = any(
        path and path[-1].lower() == "evaluator_version" and isinstance(value, str) and bool(value.strip())
        for path, value in fields
    )
    has_withheld = any(
        path
        and path[-1].lower() in {"withheld_positive_pixels", "withheld_positive_count", "withheld_positives"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
        for path, value in fields
    )
    has_score = any(
        path
        and path[-1].lower() in {"dti", "score"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
        for path, value in fields
    )
    has_ci = any(
        path
        and path[-1].lower() in {"ci95", "ci_95", "95_percent_ci"}
        and isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
                for item in value)
        and 0.0 <= float(value[0]) <= float(value[1]) <= 1.0
        for path, value in fields
    )
    return has_label and has_evaluator and has_withheld and has_score and has_ci


def _selector_approval_recorded(card: dict[str, Any]) -> bool:
    """Return true only for an explicit, frozen human-selector approval."""
    return any(
        _at(card, *path) is True
        for path in (
            ("approved_for_weekly_slot",),
            ("selector_approved",),
            ("selector", "approved"),
            ("selector", "approved_for_weekly_slot"),
        )
    )


def _metadata_is_valid(name: Any, note: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name.strip())
        and isinstance(note, str)
        and bool(note.strip())
        and len(note) <= METADATA_NOTE_LIMIT
    )


def _walk_numbers(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], float]]:
    """Yield numeric leaves in a JSON-compatible run card, excluding booleans."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_numbers(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_numbers(child, path + (str(index),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, float(value)


def _metric_name(path: tuple[str, ...]) -> str:
    """Return the nearest JSON field name for a numeric leaf/list member."""
    return next((part.lower() for part in reversed(path) if not part.isdigit()), "")


def _is_registry_uniqueness_path(path: tuple[str, ...]) -> bool:
    """Require a registry/uniqueness context, not an arbitrary feature statistic."""
    return any("registry" in part.lower() or "uniqueness" in part.lower() for part in path)


def _is_config_or_count_metric(name: str) -> bool:
    """Exclude settings and aggregate counters, while retaining dot-overlap rates."""
    return any(token in name for token in (
        "limit", "threshold", "radius", "count", "expected", "checked", "invalid", "firing",
    ))


def _threshold_observations(card: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Find Rule-1 registry measurements across documented historical schemas.

    A parent name such as ``correlation_overlap_vs_registry`` is *context*, not
    itself proof that every numeric child is a correlation or overlap rate.
    We therefore require (1) a registry/uniqueness path and (2) a metric name
    that specifically says Spearman/rho or overlap/Jaccard.  This retains fields
    such as ``post_placement_dots_max_forward_3px_overlap_real_models`` while
    excluding sibling counts like ``registry_checked`` and unrelated feature
    correlations.  Unknown/missing measurements are handled by the separate
    required-evidence gate below, rather than assumed clear.
    """
    rho: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    for path, value in _walk_numbers(card):
        name = _metric_name(path)
        if not _is_registry_uniqueness_path(path) or _is_config_or_count_metric(name):
            continue
        record = {"path": ".".join(path), "value": value}
        if "spearman" in name or "rho" in name:
            rho.append(record)
        if "overlap" in name or "jaccard" in name:
            overlap.append(record)
    return {"rho": rho, "overlap": overlap}


def _is_surface_rho_measurement(item: dict[str, Any]) -> bool:
    key = item["path"].lower()
    return any(token in key for token in ("surface", "pre_placement", "full_footprint"))


def _is_final_dot_overlap_measurement(item: dict[str, Any]) -> bool:
    key = item["path"].lower()
    # Historical cards call the completed placement ``phase2`` or put the
    # final full-footprint result under ``literal_reading``; newer cards say
    # dots/placement/final explicitly.  All are final-artifact measurements,
    # unlike phase1/pre-placement surface diagnostics.
    return any(token in key for token in (
        "dot", "placement", "final", "postwrite", "phase2", "literal_reading",
    ))


def _explicit_full_registry_certificate(card: dict[str, Any]) -> bool:
    """Accept only an explicit complete-registry certification.

    ``complete_accessible_scan`` and a count of locally downloaded rasters are
    useful diagnostics but are not a certificate that the applicable registry
    was complete.  The protocol requires a complete registry before promotion,
    so a missing declaration intentionally fails closed.
    """
    certificate_paths = (
        ("full_registry_certified",),
        ("complete_registry_certified",),
        ("registry", "full_registry_certified"),
        ("registry", "complete_registry_certified"),
        ("registry_comparison", "full_registry_certified"),
        ("registry_comparison", "complete_registry_certified"),
        ("uniqueness_audit", "full_registry_certified"),
        ("uniqueness_vs_registry", "full_registry_certified"),
    )
    return any(_at(card, *path) is True for path in certificate_paths)


def _has_recorded_duplicate(card: dict[str, Any], observations: dict[str, list[dict[str, Any]]]) -> bool:
    """Return true for a literal duplicate assertion or a Rule-1 threshold firing."""
    ua = _as_dict(card.get("uniqueness_audit"))
    rcmp = _as_dict(card.get("registry_comparison"))
    uvr = _as_dict(card.get("uniqueness_vs_registry"))
    cov = _as_dict(card.get("correlation_overlap_vs_registry"))
    reg = _as_dict(card.get("registry"))
    literal = _as_dict(rcmp.get("literal_reading"))

    declared = (
        card.get("protocol_unique") is False
        or card.get("policy_i17_literal_verdict") == "negative"
        or ua.get("template_literal_reading_duplicate") is True
        or rcmp.get("template_literal_reading_duplicate") is True
        or literal.get("duplicate") is True
        or cov.get("duplicate_stop") is True
        or reg.get("duplicate_logged") not in (None, False, "", 0)
        or (reg.get("duplicate_count") or 0) > 0
        or (reg.get("duplicate_rasters_in_checked_subset") or 0) > 0
        or (card.get("literal_reading_firings_final_dots") or 0) > 0
    )
    # Spearman is signed, so Rule 1 uses absolute correlation; forward overlap
    # is a fraction and is compared directly.  Equality remains permissible:
    # the protocol stop condition is strictly greater than the stated limits.
    observed_rule1_firing = any(abs(item["value"]) > RHO_LIMIT for item in observations["rho"]) or any(
        item["value"] > OVERLAP_LIMIT for item in observations["overlap"]
    )
    return bool(declared or observed_rule1_firing)


def _format_preflight(path: Path) -> dict[str, Any]:
    """Re-read one TIFF with the shared validator; errors are evidence, not exceptions."""
    if not path.is_file():
        return {"available": False, "all_checks_passed": False, "error": "artifact does not exist"}
    try:
        grid = load_grid(ROOT / "data" / "bridge")
        receipt = validate(path, grid.footprint, grid.catalogue)
        receipt["file"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        receipt["available"] = True
        return receipt
    except Exception as exc:  # A malformed TIFF must not crash an authorization audit.
        return {
            "available": True,
            "all_checks_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _missing_run_card(run_card_rel: str | None, reason: str) -> dict[str, Any]:
    return {
        "run_card": run_card_rel,
        "present": False,
        "error": reason,
        "recorded_verdict": None,
        "recorded_safe_to_submit": None,
        "literal_duplicate_recorded": False,
        "measured_rule1_rho_firings": [],
        "measured_rule1_overlap_firings": [],
        "surface_uniqueness_measurement_present": False,
        "final_dot_uniqueness_measurement_present": False,
        "full_registry_certified": False,
        "registry_incomplete": True,
        "holdout_declared_valid_for_promotion": False,
        "holdout_evidence_complete": False,
        "holdout_valid_for_promotion": False,
        "holdout_invalid_for_promotion": True,
        "selector_approval_recorded": False,
        "submission_metadata": {"name": None, "note": None},
        "submission_metadata_valid": False,
        "historical_promotion_conflict": False,
        "guardrail_clear": False,
    }


def audit_run_card(run_card_rel: str | None, root: Path = ROOT) -> dict[str, Any]:
    """Audit a frozen run card without mutating it in place.

    A run card is mandatory for an authorization decision.  Missing metadata,
    lack of an explicit full-registry certificate, or a missing explicit
    promotion-valid holdout field all fail closed.
    """
    if not run_card_rel:
        return _missing_run_card(None, "missing required run_card path")
    rc_path = root / run_card_rel
    if not rc_path.is_file():
        return _missing_run_card(run_card_rel, "required run_card file is absent")
    try:
        card = json.loads(rc_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return _missing_run_card(run_card_rel, f"unreadable run_card: {type(exc).__name__}: {exc}")
    if not isinstance(card, dict):
        return _missing_run_card(run_card_rel, "run_card root is not a JSON object")

    verdict = card.get("verdict")
    safe_sub = card.get("safe_to_download_and_submit") is True or card.get("ok_to_submit") is True
    observations = _threshold_observations(card)
    literal_dup = _has_recorded_duplicate(card, observations)
    # Promotion needs both parts of Rule 1: a surface-rank measurement and a
    # final-dot forward-overlap measurement.  A card cannot certify a complete
    # registry while omitting either measurement and receive an implicit pass.
    surface_measured = any(_is_surface_rho_measurement(item) for item in observations["rho"])
    final_dots_measured = any(_is_final_dot_overlap_measurement(item) for item in observations["overlap"])
    full_registry = _explicit_full_registry_certificate(card)
    # A promotion-grade comparison must be explicitly declared *and* carry the
    # evaluator, withheld-positive count, score and CI that make it auditable;
    # a positive scalar or a row in lanes.json is not a replacement for it.
    holdout_declared = card.get("holdout_valid_for_promotion") is True
    holdout_evidence_complete = _promotion_holdout_evidence_complete(card)
    holdout_valid = holdout_declared and holdout_evidence_complete
    selector_approved = _selector_approval_recorded(card)
    metadata = card_submission_metadata(card)
    metadata_valid = _metadata_is_valid(metadata["name"], metadata["note"])
    guardrail_clear = bool(
        not literal_dup
        and surface_measured
        and final_dots_measured
        and full_registry
        and holdout_valid
        and selector_approved
        and metadata_valid
    )
    conflict = bool(
        (verdict == "promote" or safe_sub)
        and (not guardrail_clear or not selector_approved)
    )
    return {
        "run_card": run_card_rel,
        "present": True,
        "recorded_verdict": verdict,
        "recorded_safe_to_submit": safe_sub,
        "literal_duplicate_recorded": literal_dup,
        "measured_rule1_rho_firings": [
            item for item in observations["rho"] if abs(item["value"]) > RHO_LIMIT
        ],
        "measured_rule1_overlap_firings": [
            item for item in observations["overlap"] if item["value"] > OVERLAP_LIMIT
        ],
        "surface_uniqueness_measurement_present": surface_measured,
        "final_dot_uniqueness_measurement_present": final_dots_measured,
        "full_registry_certified": full_registry,
        "registry_incomplete": not full_registry,
        "holdout_declared_valid_for_promotion": holdout_declared,
        "holdout_evidence_complete": holdout_evidence_complete,
        "holdout_valid_for_promotion": holdout_valid,
        "holdout_invalid_for_promotion": not holdout_valid,
        "selector_approval_recorded": selector_approved,
        "submission_metadata": metadata,
        "submission_metadata_valid": metadata_valid,
        "historical_promotion_conflict": conflict,
        "guardrail_clear": guardrail_clear,
    }


def assess(row: dict[str, Any], docs: Path, root: Path = ROOT) -> dict[str, Any]:
    """Assess one manifest row.  Every authorization prerequisite is conjunctive."""
    path = docs / row["file"]
    rc_audit = audit_run_card(row.get("run_card"), root=root)
    validator = _format_preflight(path)
    row_metadata_valid = _metadata_is_valid(row.get("name"), row.get("note"))
    card_metadata = rc_audit["submission_metadata"]
    metadata_matches_card = bool(
        row_metadata_valid
        and rc_audit["submission_metadata_valid"]
        and row.get("name") == card_metadata["name"]
        and row.get("note") == card_metadata["note"]
    )
    expected = row.get("sha256")
    measured = sha256(path) if path.is_file() else None
    checks = {
        "artifact_exists": path.is_file(),
        "sha256_matches": bool(expected and measured == expected),
        "format_valid": bool(row.get("format_valid") is True and validator.get("all_checks_passed") is True),
        "submission_metadata_valid": row_metadata_valid,
        "submission_metadata_matches_run_card": metadata_matches_card,
        "literal_protocol_unique": (
            row.get("protocol_unique") is True
            and not rc_audit["literal_duplicate_recorded"]
            and rc_audit["surface_uniqueness_measurement_present"]
            and rc_audit["final_dot_uniqueness_measurement_present"]
            and rc_audit["full_registry_certified"]
        ),
        "beats_accepted_holdout_best": (
            row.get("beats_holdout_best") is True
            and rc_audit["holdout_valid_for_promotion"]
        ),
        "scientific_verdict_promote": (
            row.get("verdict") == "promote"
            and rc_audit["recorded_verdict"] == "promote"
            and rc_audit["guardrail_clear"]
        ),
        "selector_approved": (
            row.get("selector_approved") is True
            and rc_audit["selector_approval_recorded"]
        ),
    }
    authorized = all(checks.values()) and rc_audit["guardrail_clear"]
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "lane": row["lane"],
        "file": row["file"],
        "sha256": measured,
        "validator": validator,
        "checks": checks,
        "run_card_audit": rc_audit,
        "authorized_to_submit": authorized,
        "decision": "OK TO SUBMIT" if authorized else "NOT OK TO SUBMIT",
        "failed_gates": failed,
    }


def build(lanes_path: Path = DEFAULT_LANES, out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """Build a complete release receipt from the reviewed manifest."""
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
                "superseded by the fail-closed repository release gate."
                if r["run_card_audit"]["historical_promotion_conflict"]
                else "Run card and operative release gate both fail closed."
            ),
        }
        for r in rows
        if r["run_card_audit"]["run_card"] is not None
    ]
    receipt = {
        "schema": "gems-submission-authorization-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": (
            "Authorization requires artifact existence and SHA-256 match; an independent shared-validator "
            "format PASS on the exact bytes; nonempty manifest metadata matching a frozen run card with a note "
            f"no longer than {METADATA_NOTE_LIMIT} characters; literal protocol uniqueness with recorded surface-rank "
            "and final-dot-overlap measurements plus an explicit complete-registry certificate; a promotion-valid "
            "holdout that beats the accepted best; matching promote verdicts; "
            "and explicit selector approval. Missing evidence or historical run-card conflicts fail closed."
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
        help="Exit 0 only when the gate runs cleanly and confirms zero falsely authorized files.",
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
