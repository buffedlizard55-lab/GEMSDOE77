import hashlib
import json
from pathlib import Path

from release_gate import assess, build

ROOT = Path(__file__).resolve().parents[1]


def test_published_portfolio_fails_closed(tmp_path):
    receipt = build(out=tmp_path / "authorization.json")
    assert receipt["repository_decision"] == "NOT OK TO SUBMIT"
    assert receipt["repository_authorized_file_count"] == 0
    assert receipt["artifacts"]
    assert all(not row["authorized_to_submit"] for row in receipt["artifacts"])
    assert all("literal_protocol_unique" in row["failed_gates"] for row in receipt["artifacts"])
    assert all(row["checks"]["artifact_exists"] for row in receipt["artifacts"])
    assert all(row["checks"]["sha256_matches"] for row in receipt["artifacts"])

    reconciliations = {r["run_card"]: r for r in receipt["historical_run_card_reconciliation"]}
    assert reconciliations["evidence/run_card_h77d.json"]["historical_promotion_conflict"] is True
    assert reconciliations["evidence/h77/run_card.json"]["historical_promotion_conflict"] is True
    assert reconciliations["evidence/h78_run_card.json"]["historical_promotion_conflict"] is False
    assert reconciliations["evidence/run_card.json"]["historical_promotion_conflict"] is False


def test_format_pass_alone_never_authorizes(tmp_path):
    artifact = tmp_path / "candidate.tif"
    artifact.write_bytes(b"unique-test-artifact")
    row = {
        "lane": "test",
        "file": artifact.name,
        "format_valid": True,
        "protocol_unique": False,
        "beats_holdout_best": True,
        "verdict": "promote",
        "selector_approved": True,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    result = assess(row, tmp_path)
    assert not result["authorized_to_submit"]
    assert result["decision"] == "NOT OK TO SUBMIT"
    assert result["checks"]["format_valid"]
    assert not result["checks"]["literal_protocol_unique"]


def test_missing_selector_decision_fails_closed():
    row = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"][0]
    result = assess(row, ROOT / "docs")
    assert result["checks"]["selector_approved"] is False
    assert result["authorized_to_submit"] is False


def test_run_card_schema_guardrail_blocks_false_promotion(tmp_path):
    artifact = tmp_path / "candidate.tif"
    artifact.write_bytes(b"unique-test-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    for rc_rel in (
        "evidence/h77/run_card.json",
        "evidence/run_card_h77d.json",
        "evidence/run_card.json",
    ):
        row = {
            "lane": f"test-{rc_rel}",
            "file": artifact.name,
            "run_card": rc_rel,
            "format_valid": True,
            "protocol_unique": True,
            "beats_holdout_best": True,
            "verdict": "promote",
            "selector_approved": True,
            "sha256": digest,
        }
        result = assess(row, tmp_path, root=ROOT)
        assert result["authorized_to_submit"] is False, rc_rel
        assert result["decision"] == "NOT OK TO SUBMIT", rc_rel
        assert result["run_card_audit"]["guardrail_clear"] is False, rc_rel
