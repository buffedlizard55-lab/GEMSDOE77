import hashlib
import json
import shutil
from pathlib import Path

from release_gate import assess, audit_run_card, build

ROOT = Path(__file__).resolve().parents[1]
VALID_TIF = ROOT / "docs" / "downloads" / "gems77-h77l-thk-flank-6b24305363fa-RESEARCH-ONLY.tif"


def _copy_valid_tif(tmp_path):
    artifact = tmp_path / "candidate.tif"
    shutil.copyfile(VALID_TIF, artifact)
    return artifact


def test_published_portfolio_fails_closed(tmp_path):
    receipt = build(out=tmp_path / "authorization.json")
    assert receipt["repository_decision"] == "NOT OK TO SUBMIT"
    assert receipt["repository_authorized_file_count"] == 0
    assert receipt["artifacts"]
    assert all(not row["authorized_to_submit"] for row in receipt["artifacts"])
    assert all("literal_protocol_unique" in row["failed_gates"] for row in receipt["artifacts"])
    assert all(row["checks"]["artifact_exists"] for row in receipt["artifacts"])
    assert all(row["checks"]["sha256_matches"] for row in receipt["artifacts"])
    # Release status is based on a re-read of exact TIFF bytes, not an unchecked
    # boolean in lanes.json. Every retained research file happens to pass this
    # local format preflight, but none is consequently authorized.
    assert all(row["validator"]["all_checks_passed"] for row in receipt["artifacts"])
    assert all(row["checks"]["format_valid"] for row in receipt["artifacts"])
    assert all(row["checks"]["submission_metadata_valid"] for row in receipt["artifacts"])
    assert all(row["checks"]["submission_metadata_matches_run_card"] for row in receipt["artifacts"])
    assert all(row["run_card_audit"]["present"] for row in receipt["artifacts"])
    assert all(not row["run_card_audit"]["full_registry_certified"] for row in receipt["artifacts"])

    reconciliations = {r["run_card"]: r for r in receipt["historical_run_card_reconciliation"]}
    assert reconciliations["evidence/run_card_h77d.json"]["historical_promotion_conflict"] is True
    assert reconciliations["evidence/h77/run_card.json"]["historical_promotion_conflict"] is True
    assert reconciliations["evidence/h78_run_card.json"]["historical_promotion_conflict"] is False
    assert reconciliations["evidence/run_card.json"]["historical_promotion_conflict"] is False


def test_published_receipt_includes_each_current_lane_and_sha(tmp_path):
    generated = build(out=tmp_path / "fresh.json")
    published = json.loads((ROOT / "docs/data/submission-authorization.json").read_text())
    assert generated["schema"] == published["schema"]
    assert generated["repository_decision"] == published["repository_decision"] == "NOT OK TO SUBMIT"
    assert generated["repository_authorized_file_count"] == published["repository_authorized_file_count"] == 0
    assert len(generated["artifacts"]) == len(published["artifacts"]) == len(
        json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"]
    )
    for now, saved in zip(generated["artifacts"], published["artifacts"]):
        assert now == saved


def test_format_pass_alone_never_authorizes_without_a_run_card(tmp_path):
    artifact = _copy_valid_tif(tmp_path)
    row = {
        "lane": "test",
        "file": artifact.name,
        "format_valid": True,
        "protocol_unique": True,
        "beats_holdout_best": True,
        "verdict": "promote",
        "selector_approved": True,
        "name": "valid-local-metadata",
        "note": "Format passed, but the mandatory run card is absent.",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    result = assess(row, tmp_path)
    assert not result["authorized_to_submit"]
    assert result["decision"] == "NOT OK TO SUBMIT"
    assert result["checks"]["format_valid"]
    assert result["checks"]["submission_metadata_valid"]
    assert not result["checks"]["submission_metadata_matches_run_card"]
    assert not result["run_card_audit"]["present"]
    assert result["run_card_audit"]["registry_incomplete"]


def test_complete_explicit_synthetic_card_is_the_only_authorization_path(tmp_path):
    """A future lane needs every frozen release prerequisite, not a bypass."""
    artifact = _copy_valid_tif(tmp_path)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    card_path = tmp_path / "evidence" / "candidate.json"
    card_path.parent.mkdir()
    card_path.write_text(json.dumps({
        "verdict": "promote",
        "submission_name": "synthetic-cleared-candidate",
        "submission_note": "Synthetic fixture for every explicit release prerequisite.",
        "full_registry_certified": True,
        "holdout_valid_for_promotion": True,
        "approved_for_weekly_slot": True,
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": "synthetic-evaluator-v1",
            "withheld_positive_pixels": 100,
            "dti": 0.20,
            "ci95": [0.15, 0.25],
        },
        "uniqueness_vs_registry": {
            "surface_max_spearman": 0.12,
            "post_placement_dots_max_forward_3px_overlap": 0.22,
        },
    }))
    row = {
        "lane": "synthetic-cleared-lane",
        "file": artifact.name,
        "run_card": "evidence/candidate.json",
        "format_valid": True,
        "protocol_unique": True,
        "beats_holdout_best": True,
        "verdict": "promote",
        "selector_approved": True,
        "name": "synthetic-cleared-candidate",
        "note": "Synthetic fixture for every explicit release prerequisite.",
        "sha256": digest,
    }
    result = assess(row, tmp_path, root=tmp_path)
    assert result["authorized_to_submit"] is True
    assert result["decision"] == "OK TO SUBMIT"
    assert not result["failed_gates"]


def test_missing_selector_decision_fails_closed():
    row = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"][0]
    result = assess(row, ROOT / "docs")
    assert result["checks"]["selector_approved"] is False
    assert result["authorized_to_submit"] is False


def test_final_dot_forward_overlap_named_with_dots_stops_historical_promote_lane():
    """Rule 1 must not be bypassed merely because the field includes `dots`."""
    audit = audit_run_card("evidence/h77/run_card.json", root=ROOT)
    assert audit["surface_uniqueness_measurement_present"] is True
    assert audit["final_dot_uniqueness_measurement_present"] is True
    assert audit["literal_duplicate_recorded"] is True
    assert any(
        item["path"] == "uniqueness_vs_registry.post_placement_dots_max_forward_3px_overlap_real_models"
        and item["value"] > 0.70
        for item in audit["measured_rule1_overlap_firings"]
    )


def test_rule1_uses_absolute_spearman_and_requires_both_measurements(tmp_path):
    card_path = tmp_path / "evidence" / "candidate.json"
    card_path.parent.mkdir()
    card_path.write_text(json.dumps({
        "submission_name": "synthetic-name",
        "submission_note": "Synthetic card exercises signed surface and final-dot Rule 1 checks.",
        "uniqueness_vs_registry": {
            "surface_max_spearman": -0.91,
            "post_placement_dots_max_forward_3px_overlap": 0.71,
        },
    }))
    audit = audit_run_card("evidence/candidate.json", root=tmp_path)
    assert audit["surface_uniqueness_measurement_present"]
    assert audit["final_dot_uniqueness_measurement_present"]
    assert audit["literal_duplicate_recorded"] is True
    assert audit["measured_rule1_rho_firings"] == [
        {"path": "uniqueness_vs_registry.surface_max_spearman", "value": -0.91}
    ]
    assert audit["measured_rule1_overlap_firings"] == [
        {"path": "uniqueness_vs_registry.post_placement_dots_max_forward_3px_overlap", "value": 0.71}
    ]


def test_run_card_schema_guardrail_blocks_false_promotion(tmp_path):
    artifact = _copy_valid_tif(tmp_path)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    cards = {
        "evidence/h77/run_card.json": (
            "gems77-step-propensity-submod300",
            "GEMSDOE77: HW/FW dip-slip step tensor + V-LBCO crack-tip mechanics + Radon multi-scale ensemble + flank-zeroed 300m submod",
        ),
        "evidence/run_card_h77d.json": (
            "gems77-h77d-hydrotect-submod-37654dots-20261021-0176be7f-zeros",
            "H77D zero-SGMC scarp+hydro+LOCO mechanics OOF + 300m submodular (37654 dots); HOLDOUT-DTI 0.1501 (s30) / 0.1403 (s20) vs 0.1093 prior best",
        ),
        "evidence/run_card.json": (
            "gems77-h77-urank-curvature-0d868dc271bb-research",
            "H77 selective-U rank curvature; research-only, literal registry gate failed; do not submit.",
        ),
    }
    for rc_rel, (name, note) in cards.items():
        row = {
            "lane": f"test-{rc_rel}",
            "file": artifact.name,
            "run_card": rc_rel,
            "format_valid": True,
            "protocol_unique": True,
            "beats_holdout_best": True,
            "verdict": "promote",
            "selector_approved": True,
            "name": name,
            "note": note,
            "sha256": digest,
        }
        result = assess(row, tmp_path, root=ROOT)
        assert result["authorized_to_submit"] is False, rc_rel
        assert result["decision"] == "NOT OK TO SUBMIT", rc_rel
        assert result["checks"]["submission_metadata_matches_run_card"], rc_rel
        assert result["checks"]["selector_approved"] is False, rc_rel
        assert result["run_card_audit"]["selector_approval_recorded"] is False, rc_rel
        assert result["run_card_audit"]["guardrail_clear"] is False, rc_rel


def test_run_card_needs_explicit_registry_and_holdout_certificates(tmp_path):
    card_path = tmp_path / "evidence" / "candidate.json"
    card_path.parent.mkdir()
    card_path.write_text(json.dumps({
        "verdict": "promote",
        "holdout_valid_for_promotion": True,
        "submission_name": "synthetic-name",
        "submission_note": "A synthetically complete-looking card that omits two mandatory certificates.",
    }))
    audit = audit_run_card("evidence/candidate.json", root=tmp_path)
    assert audit["present"]
    assert audit["submission_metadata_valid"]
    assert not audit["full_registry_certified"]
    assert not audit["surface_uniqueness_measurement_present"]
    assert not audit["final_dot_uniqueness_measurement_present"]
    assert audit["registry_incomplete"]
    assert audit["holdout_declared_valid_for_promotion"]
    assert not audit["holdout_evidence_complete"]
    assert not audit["holdout_valid_for_promotion"]
    assert audit["holdout_invalid_for_promotion"]
    assert not audit["guardrail_clear"]


def test_manifest_metadata_is_unique_and_portal_length_safe():
    lanes = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"]
    assert all(row.get("name") and row.get("note") and row.get("run_card") for row in lanes)
    assert all(len(row["note"]) <= 140 for row in lanes)
    assert len({row["name"] for row in lanes}) == len(lanes)
