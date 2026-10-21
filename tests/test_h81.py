"""H81 slip-cone lane: frozen-transform unit checks + artifact audit.

Unit checks run on synthetic graphs (no data needed). The artifact audit runs
only after evidence/h81_run_card.json exists (i.e. after E3).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _chain_deg1(segs):
    """Reference re-implementation of the strand-tip rule for tiny graphs."""
    from collections import defaultdict
    adj = defaultdict(list)
    for i, (a, b) in enumerate(segs):
        adj[a].append((b, i))
        adj[b].append((a, i))
    seen, tips = set(), []
    for node in adj:
        if node in seen:
            continue
        stack, nodes = [node], []
        seen.add(node)
        while stack:
            cur = stack.pop()
            nodes.append(cur)
            for nb, _i in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        tips.extend(v for v in nodes if len(adj[v]) == 1)
    return tips


def test_strand_tip_rule_excludes_joints_and_loops():
    # straight chain of 3 segments: 2 tips, joints excluded
    assert sorted(_chain_deg1([((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))])) == [
        (0, 0), (3, 0)]
    # Y splay: 3 tips (center joint excluded)
    assert len(_chain_deg1([((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 0), (1, -1))])) == 3
    # closed loop: no tips
    assert _chain_deg1([((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 0))]) == []
    # two components separated by a gap: 4 tips (gap-bounded strand ends kept)
    segs = [((0, 0), (1, 0)), ((5, 0), (6, 0))]
    assert len(_chain_deg1(segs)) == 4


def test_cone_geometry_half_angle_and_decay():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "vendor"))
    from h81_pipeline import rasterize_cones, CONE_LEN_PX
    tip = {"r": 50.0, "c": 50.0, "dr": 0.0, "dc": 1.0}  # pointing +col
    out = rasterize_cones([tip], np.array([1.0], np.float32), (100, 100))
    assert out[50, 50] == 1.0  # apex full weight
    assert out[50, 50 + CONE_LEN_PX] == 0.0  # linear decay hits 0 at L
    assert out[50, 50 + CONE_LEN_PX // 2] > 0.4  # mid-cone positive
    assert out[50, 50 - 5] == 0.0  # nothing behind the tip
    # 30-degree half-angle: at distance 10 along axis, half-width = 10*tan(30)
    import math
    hw = int(10 * math.tan(math.radians(30)))
    assert out[50, 60] > 0
    assert out[50 + hw + 2, 60] == 0.0
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_preregistration_frozen_numbers():
    pre = json.loads((ROOT / "registry/preregistration_h81.json").read_text())
    assert pre["emission"]["budget"] == 37654
    assert pre["folds"]["seed"] == 20
    assert pre["fixed_transform"]["cone"] == "half-angle 30 deg, length 20 px (2.0 km), linear decay (1 - d/20); pixel value = max over tips of weight*decay, float32"


def test_h81_artifact_audit():
    card_p = ROOT / "evidence/h81_run_card.json"
    if not card_p.exists():
        import pytest
        pytest.skip("H81 E3 not run yet")
    card = json.loads(card_p.read_text())
    tif = ROOT / "submissions" / f"{card['submission_name']}.tif"
    assert tif.is_file()
    assert hashlib.sha256(tif.read_bytes()).hexdigest() == card["raster_sha256"]
    assert card["ok_to_download"] is True
    assert card["ok_to_submit"] is False
    assert card["submission_slots_used"] == 0
    assert len(card["submission_note"]) <= 140
    assert card["holdout_dti"]["label"] == "HOLDOUT-DTI"
    assert card["holdout_dti"]["evaluator_version"] == "gems57-pooled-hide-v2"
    assert card["validator_output"]["all_checks_passed"] is True
    page = (ROOT / "docs/h81.html").read_text()
    assert "OK to download" in page
    assert "NOT OK to submit" in page or "NOT OK TO SUBMIT" in page
    assert card["raster_sha256"] in page
