"""Preflight is a rejection witness, never permission to submit."""
import json
from pathlib import Path

import numpy as np
import pytest

from preflight_registry import audit, domain_proof

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
