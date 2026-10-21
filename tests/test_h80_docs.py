from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_h80_page_separates_reported_score_from_causal_evidence():
    page = (ROOT / "docs/h80.html").read_text()
    assert "NOT OK TO SUBMIT" in page
    assert "0.2778 remains <b>OWNER-REPORT</b>" in page
    assert "no organizer receipt here links these bytes" in page
    assert "not a controlled causal result" in page
    assert "HOLDOUT-DTI" in page
    assert "gems57-pooled-hide-v2" in page
    assert "11,319 withheld positives" in page
    assert "95% CI [0.08818, 0.10325]" in page
    assert "Why potentially missing from the catalogue (hypothesis)" in page


def test_h80_published_hypotheses_mirror_is_current_and_qualified():
    source = ROOT / "research/h80_hypotheses.md"
    published = ROOT / "docs/research/h80_hypotheses.md"
    assert source.is_file() and published.is_file()
    assert source.read_bytes() == published.read_bytes()
    text = published.read_text()
    assert "Post-run clarification" in text
    assert "not causal or label-proven results" in text
    assert "no controlled experiment here establishes" in text
