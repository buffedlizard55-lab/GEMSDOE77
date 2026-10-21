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
        "sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
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
