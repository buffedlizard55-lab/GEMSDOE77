import hashlib
import json
import shutil

import pytest
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


def test_published_receipt_includes_each_current_lane_and_sha(tmp_path):
    generated=build(out=tmp_path / 'fresh.json')
    published=json.loads((ROOT/'docs/data/submission-authorization.json').read_text())
    assert generated['schema']==published['schema']
    assert generated['repository_decision']==published['repository_decision']=='NOT OK TO SUBMIT'
    assert generated['repository_authorized_file_count']==published['repository_authorized_file_count']==0
    assert len(generated['artifacts'])==len(published['artifacts'])==len(
        json.loads((ROOT/'docs/data/lanes.json').read_text())['lanes'])
    for now,saved in zip(generated['artifacts'],published['artifacts']):
        assert now==saved


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


def approval_row(tmp_path):
    artifact = tmp_path / "candidate.tif"
    source = json.loads((ROOT / "docs/data/lanes.json").read_text())["lanes"][0]
    shutil.copyfile(ROOT / "docs" / source["file"], artifact)
    shutil.copytree(ROOT / "data/bridge", tmp_path / "data/bridge")
    (tmp_path / "registry").mkdir()
    shutil.copyfile(ROOT / "registry/data_manifest.json", tmp_path / "registry/data_manifest.json")
    return {
        "lane": "approval-regression", "file": artifact.name,
        "format_valid": True, "protocol_unique": True,
        "beats_holdout_best": True, "verdict": "promote",
        "selector_approved": True,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def test_missing_run_card_blocks_otherwise_green_manifest(tmp_path):
    row = approval_row(tmp_path)
    result = assess(row, tmp_path, root=tmp_path)
    assert not result["authorized_to_submit"]
    assert "run_card_present" in result["failed_gates"]
    assert not result["run_card_audit"]["guardrail_clear"]


def test_run_card_verdict_must_agree_with_manifest(tmp_path):
    row = approval_row(tmp_path)
    row["run_card"] = "card.json"
    for verdict in ("negative", None, "PROMOTE", True):
        (tmp_path / "card.json").write_text(json.dumps({"verdict": verdict}))
        result = assess(row, tmp_path, root=tmp_path)
        assert not result["authorized_to_submit"]
        assert "scientific_verdict_promote" in result["failed_gates"]


def test_malformed_run_card_fails_closed_without_stale_receipt(tmp_path):
    row = approval_row(tmp_path)
    row["run_card"] = "card.json"
    for text in ('{broken', '[]', 'null', '42'):
        (tmp_path / "card.json").write_text(text)
        result = assess(row, tmp_path, root=tmp_path)
        assert not result["authorized_to_submit"]
        assert result["run_card_audit"]["read_error"]
        assert not result["checks"]["run_card_present"]


def test_consistent_promotion_can_pass_recorded_decision_gate(tmp_path):
    # The gate checks bytes and binding, but still consumes recorded science/selector decisions.
    row = approval_row(tmp_path)
    row["run_card"] = "card.json"
    (tmp_path / "card.json").write_text(json.dumps({
        "verdict": "promote", "raster_sha256": row["sha256"],
        "protocol_unique": True, "holdout_valid_for_promotion": True,
        "beats_holdout_best": True, "approved_for_weekly_slot": True,
        "registry": {"full_registry_certified": True},
    }))
    assert assess(row, tmp_path, root=tmp_path)["authorized_to_submit"]
    row["selector_approved"] = False
    assert not assess(row, tmp_path, root=tmp_path)["authorized_to_submit"]


@pytest.mark.parametrize("sha", [None, "0" * 64, True])
def test_unbound_or_other_artifact_card_cannot_authorize(tmp_path, sha):
    row = approval_row(tmp_path)
    row["run_card"] = "card.json"
    (tmp_path / "card.json").write_text(json.dumps({"verdict": "promote", "raster_sha256": sha}))
    result = assess(row, tmp_path, root=tmp_path)
    assert not result["authorized_to_submit"]
    assert not result["checks"]["run_card_artifact_matches"]


def test_matching_hash_cannot_authorize_fake_tiff(tmp_path):
    row = approval_row(tmp_path)
    artifact = tmp_path / row["file"]
    artifact.write_bytes(b"not a GeoTIFF")
    row["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    row["run_card"] = "card.json"
    (tmp_path / "card.json").write_text(json.dumps({"verdict": "promote", "raster_sha256": row["sha256"]}))
    result = assess(row, tmp_path, root=tmp_path)
    assert result["checks"]["sha256_matches"]
    assert result["checks"]["run_card_artifact_matches"]
    assert not result["checks"]["format_readback_passed"]
    assert not result["authorized_to_submit"]


@pytest.mark.parametrize("payload", [
    {"registry": [1]}, {"registry": "complete"},
    {"correlation_overlap_vs_registry": {"phase": {"dot_duplicate_count": "zero"}}},
    {"registry": {"complete": "true"}},
    {"registry": {"surface_max_forward_overlap_3px": float("nan")}},
    {"registry": {"surface_max_forward_overlap_3px": float("inf")}},
    {"registry": {"duplicate_count": True}},
])
def test_malformed_nested_guardrails_are_denied_not_crashes(tmp_path, payload):
    row = approval_row(tmp_path)
    row["run_card"] = "card.json"
    payload.update(verdict="promote", raster_sha256=row["sha256"])
    (tmp_path / "card.json").write_text(json.dumps(payload))
    result = assess(row, tmp_path, root=tmp_path)
    assert not result["authorized_to_submit"]
    assert result["run_card_audit"]["read_error"]


def test_modified_reference_cannot_authorize(tmp_path):
    row = approval_row(tmp_path)
    row["run_card"] = "card.json"
    (tmp_path / "card.json").write_text(json.dumps({"verdict": "promote", "raster_sha256": row["sha256"]}))
    (tmp_path / "data/bridge/sample_submission.tif").write_bytes(b"wrong reference")
    result = assess(row, tmp_path, root=tmp_path)
    assert not result["authorized_to_submit"]
    assert "reference hash mismatch" in result["format_readback"]["error"]


@pytest.mark.parametrize("value", [-9999.0, 1.1, float("nan"), float("inf")])
def test_green_manifest_cannot_override_actual_pixel_range(tmp_path, value):
    import rasterio
    from rasterio.windows import Window
    import numpy as np
    row = approval_row(tmp_path)
    path = tmp_path / row["file"]
    with rasterio.open(path, "r+") as dst:
        dst.write(np.array([[value]], dtype="float32"), 1, window=Window(0, 0, 1, 1))
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row["run_card"] = "card.json"
    (tmp_path / "card.json").write_text(json.dumps({"verdict":"promote", "raster_sha256":row["sha256"]}))
    result = assess(row, tmp_path, root=tmp_path)
    assert result["checks"]["sha256_matches"]
    assert result["checks"]["run_card_artifact_matches"]
    assert not result["checks"]["format_readback_passed"]
    assert not result["authorized_to_submit"]


def test_matching_card_hash_alone_is_not_positive_science_evidence(tmp_path):
    row = approval_row(tmp_path)
    row["run_card"] = "card.json"
    (tmp_path / "card.json").write_text(json.dumps({"verdict":"promote", "raster_sha256":row["sha256"]}))
    result = assess(row, tmp_path, root=tmp_path)
    assert result["checks"]["format_readback_passed"]
    assert result["checks"]["run_card_artifact_matches"]
    assert not result["checks"]["explicit_run_card_promotion_evidence"]
    assert not result["authorized_to_submit"]


def test_malformed_nested_guardrails_fail_closed(tmp_path):
    row = approval_row(tmp_path)
    row['run_card'] = 'card.json'
    payloads = [
        {'registry': 'complete'},
        {'uniqueness_audit': []},
        {'registry_comparison': {'literal_reading': 'clear'}},
        {'registry': {'duplicate_count': '0'}},
        {'correlation_overlap_vs_registry': {'final_dots_max_forward_overlap_3px': float('nan')}},
        {'registry': {'surface_max_forward_overlap_3px': float('inf')}},
        {'registry': {'surface_max_forward_overlap_3px': -1}},
        {'registry': {'surface_max_forward_overlap_3px': 1.1}},
        {'registry': {'duplicate_count': False}},
    ]
    for payload in payloads:
        (tmp_path / 'card.json').write_text(json.dumps({'verdict': 'promote', **payload}))
        result = assess(row, tmp_path, root=tmp_path)
        assert not result['authorized_to_submit'], payload
        assert result['run_card_audit']['read_error'], payload
