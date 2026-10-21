"""Synthetic checks for the new-physics transforms (run before trusting them)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems77.features77 import phase_congruency, pseudogravity  # noqa: E402


def test_pc_peaks_at_step_edge():
    rng = np.random.default_rng(1)
    im = np.zeros((128, 128), np.float64)
    im[:, 32:96] = 10.0  # two step edges; periodic-clean (no FFT wrap edge)
    im += rng.normal(0, 0.3, im.shape)
    pc = phase_congruency(im)
    row = pc[64, :]
    for edge in (32, 96):
        col = int(np.argmax(row[edge - 6:edge + 7])) + edge - 6
        assert abs(col - edge) <= 2, f"PC peak at col {col}, expected ~{edge}"
    assert row[32] > row[16] + 0.2 and row[32] > row[64] + 0.2, "edge PC must exceed flat areas"
    assert pc.min() >= 0.0 and pc.max() <= 1.0


def test_pc_low_on_pure_noise():
    rng = np.random.default_rng(2)
    pc = phase_congruency(rng.normal(0, 1, (128, 128)))
    assert float(np.median(pc)) < 0.35, f"noise median PC {np.median(pc):.3f} too high"


def test_pseudogravity_smooths_and_preserves_step_sign():
    im = np.zeros((128, 128), np.float32)
    im[:, 32:96] = 100.0
    pg = pseudogravity(im)
    # vertical integration turns a box anomaly into a smooth high over the box
    assert pg[64, 64] > pg[64, 32] > pg[64, 8]


if __name__ == "__main__":
    test_pc_peaks_at_step_edge(); print("PASS pc step edge")
    test_pc_low_on_pure_noise(); print("PASS pc noise floor")
    test_pseudogravity_smooths_and_preserves_step_sign(); print("PASS pseudogravity ramp")
