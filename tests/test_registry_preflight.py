"""Preflight is a rejection witness, never permission to submit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from preflight_registry import audit, domain_proof, verify_template_pin

ROOT = Path(__file__).resolve().parents[1]


def test_universal_proof_covers_every_nonempty_subset():
    allowed = np.ones((9, 9), dtype=bool)
    witness = np.ones((9, 9), dtype=float) * 0.00001
    result = domain_proof(allowed, witness)
    assert result["universal_overlap_blocker"]
    assert result["nonempty_subset_forward_overlap_lower_bound"] == 1
    assert result["uncovered_allowed_pixels"] == 0


def test_uses_euclidean_3px_inclusive_not_square_dilation():
    allowed = np.zeros((9, 9), dtype=bool)
    allowed[4, 7] = True  # exactly three pixels, included
    witness = np.zeros((9, 9))
    witness[4, 4] = 1
    assert domain_proof(allowed, witness)["universal_overlap_blocker"]
    allowed[7, 7] = True  # square radius three but Euclidean distance > three
    result = domain_proof(allowed, witness)
    assert not result["universal_overlap_blocker"]
    assert result["uncovered_allowed_pixels"] == 1
    assert result["nonempty_subset_forward_overlap_lower_bound"] is None


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_only_finite_positive_witness_support_counts(value):
    result = domain_proof(np.ones((9, 9), bool), np.full((9, 9), value))
    assert not result["universal_overlap_blocker"]
    assert result["allowed_pixels_within_3px"] == 0


def test_empty_allowed_domain_is_not_vacuous_clearance_or_rejection():
    result = domain_proof(np.zeros((5, 5), bool), np.ones((5, 5)))
    assert result["allowed_pixels"] == 0
    assert not result["universal_overlap_blocker"]
    assert result["nonempty_subset_forward_overlap_lower_bound"] is None


def test_invalid_grid_is_rejected():
    with pytest.raises(ValueError, match="2D grid"):
        domain_proof(np.ones((2, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="2D grid"):
        domain_proof(np.ones(2), np.ones(2))


def test_no_blocker_proof_never_means_unique():
    result = domain_proof(np.ones((11, 11), bool), np.eye(11))
    assert not result["universal_overlap_blocker"]
    assert "unique" not in result


def test_declared_additive_template_amendment_uses_its_local_pin(tmp_path):
    path = tmp_path / "vendor/gems57/uniqueness.py"
    path.parent.mkdir(parents=True)
    path.write_text("upstream plus declared reporting helpers")
    local_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    pin = {
        "files": {"uniqueness.py": "upstream-sha"},
        "local_amendments": {
            "uniqueness.py": {
                "upstream_sha256": "upstream-sha",
                "local_sha256": local_sha,
                "measurement_semantics_unchanged": True,
            }
        },
    }
    verify_template_pin(tmp_path, pin)
    pin["local_amendments"]["uniqueness.py"]["upstream_sha256"] = "wrong-upstream"
    with pytest.raises(ValueError, match="invalid declared shared-template amendment"):
        verify_template_pin(tmp_path, pin)


def test_declared_list_template_amendment_is_hash_pinned_not_policy_clearance(tmp_path):
    path = tmp_path / "vendor/gems57/uniqueness.py"
    path.parent.mkdir(parents=True)
    path.write_text("upstream plus a documented local policy amendment")
    local_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    pin = {
        "files": {"uniqueness.py": "upstream-sha"},
        "local_amendments": [{
            "file": "uniqueness.py",
            "irregularity": "IR-H85-01",
            "upstream_sha256": "upstream-sha",
            "local_sha256": local_sha,
            "change": "documented local policy amendment",
            "tests": "tests/test_h85.py",
        }],
    }
    verify_template_pin(tmp_path, pin)
    pin["local_amendments"][0]["upstream_sha256"] = "wrong-upstream"
    with pytest.raises(ValueError, match="invalid declared shared-template amendment"):
        verify_template_pin(tmp_path, pin)


def test_input_pin_failure_precedes_any_proof(tmp_path):
    # Hash failures stop before opening arbitrary/untrusted TIFF input bytes.
    (tmp_path / "registry").mkdir()
    (tmp_path / "vendor/gems57").mkdir(parents=True)
    (tmp_path / "registry/template_pin.json").write_text(json.dumps({"files": {"metric.py": "bad"}}))
    (tmp_path / "vendor/gems57/metric.py").write_text("not the pinned template")
    with pytest.raises(ValueError, match="shared-template hash mismatch"):
        audit(tmp_path)


def test_published_preflight_is_rejection_only():
    result = json.loads((ROOT / "evidence/registry_preflight.json").read_text())
    assert result["decision"] == "STOP_UNIVERSAL_DUPLICATE"
    assert result["proof"]["nonempty_subset_forward_overlap_lower_bound"] == 1
    assert result["authorized_to_submit"] is False
    assert result["experiments_run"] == result["weekly_slots_used"] == 0
    assert result["new_raster_generated"] is False
    pins = json.loads((ROOT / "registry/witness_sources.json").read_text())
    assert result["witness_source"] == pins["dense_witness"]


def test_cli_input_failure_replaces_a_stale_receipt(tmp_path, monkeypatch):
    import preflight_registry
    output = tmp_path / "receipt.json"
    output.write_text('{"decision":"OLD_RESULT"}')
    monkeypatch.setattr("sys.argv", ["preflight_registry.py", "--out", str(output)])
    def broken():
        raise ValueError("input pin mismatch")
    monkeypatch.setattr(preflight_registry, "audit", broken)
    assert preflight_registry.main() == 3
    result = json.loads(output.read_text())
    assert result["decision"] == "STOP_INPUT_ERROR"
    assert result["authorized_to_submit"] is False
    assert result["error"] == "input pin mismatch"
