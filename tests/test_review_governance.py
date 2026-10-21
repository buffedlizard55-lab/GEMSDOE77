from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs"


def test_all_start_pages_fail_closed_after_i17():
    """Historic scientific promotion must not be rendered as current upload clearance."""
    pages = [
        DOC / "index.html",
        DOC / "executive-summary.html",
        DOC / "h77d.html",
        DOC / "h77d-executive-summary.html",
        DOC / "h77-step.html",
        DOC / "h77-step-executive-summary.html",
        DOC / "urank.html",
    ]
    for path in pages:
        text = path.read_text()
        assert "Not OK to submit" in text or "NOT CLEARED TO SUBMIT" in text, path
        assert "PROMOTE (HOLDOUT CHAMPION)" not in text, path
        assert "recommended for drivendata upload" not in text.lower(), path
        assert "once confirmed by the human submitter" not in text.lower(), path


def test_next_lane_slate_is_research_only_and_discloses_access_limits():
    slate = (ROOT / "research" / "next-lane-candidates.md").read_text()
    assert "not executed" in slate.lower()
    assert "not viable yet" in slate.lower()
    assert "USGS 3DEP" in slate and "USGS ComCat" in slate
    assert "Landsat Collection 2" in slate
    assert "No score is forecast" in slate
    assert "No competition slot is used" in slate


def test_public_run_card_language_distinguishes_local_checks_from_acceptance():
    for path in [DOC / "index.html", DOC / "executive-summary.html", DOC / "h77d-executive-summary.html"]:
        text = path.read_text().lower()
        assert "not cleared to submit" in text or "not ok to submit" in text, path
    source = (ROOT / "scripts" / "verify_all.py").read_text()
    assert "BRIDGE / name" in source
    assert "Prefer the restored raw mirror" in source
