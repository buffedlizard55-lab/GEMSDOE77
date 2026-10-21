"""Synthetic markup fixtures: no authenticated competition score is implied."""
import io
import json
from datetime import datetime, timezone

import public_leaderboard as lb
import pytest


HTML = """<!doctype html><title>GEMS Prize</title>
<table><thead><tr><th>Rank</th><th>Team members</th><th>Participant</th>
<th>Best<br> public<br> DW-Tversky<br>(in descending order)</th><th>Shared work</th></tr></thead>
<tbody><tr><td>#1</td><td></td><td>Team One<br>23 submissions</td><td>0.4413</td><td></td></tr>
<tr><td>#2</td><td></td><td>Other</td><td>0.3904</td><td></td></tr></tbody></table>"""


class FakeResponse(io.BytesIO):
    headers = {"Content-Type": "text/html; charset=utf-8"}

    def __init__(self, html=HTML, url=lb.URL):
        super().__init__(html.encode("utf-8"))
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_exact_first_public_row_not_a_file_receipt():
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    observation = lb.observe(lambda request, timeout: FakeResponse(), now=now)
    assert observation["status"] == "retrieved"
    assert observation["public_dti"] == pytest.approx(0.4413)
    assert observation["rank"] == 1
    assert observation["participant_row"] == "Team One 23 submissions"
    assert observation["score_is_submission_page_receipt"] is False
    assert observation["submission_name"] is None and observation["raster_sha256"] is None
    assert observation["evidence_class"].startswith("PUBLIC-LEADERBOARD OBSERVATION")
    assert observation["checked_utc"] == "2026-10-21T00:00:00+00:00"
    json.dumps(observation, allow_nan=False)


@pytest.mark.parametrize("page", [
    "<h1>Login</h1>",
    HTML.replace("0.4413", "1.2"),
    HTML.replace("0.4413", "NaN"),
    HTML.replace("DW-Tversky", "Another column"),
    HTML.replace("<tr><td>#2</td>", "<tr><td>#1</td>"),
])
def test_ambiguous_markup_never_emits_a_score(page):
    observation = lb.observe(lambda request, timeout: FakeResponse(page))
    assert observation["status"] == "unavailable"
    assert observation["public_dti"] is None
    assert observation["score_is_submission_page_receipt"] is False


def test_redirect_or_down_site_fails_closed():
    redirected = lb.observe(lambda request, timeout: FakeResponse(url="https://www.drivendata.org/accounts/login/"))
    assert redirected["status"] == "unavailable" and redirected["public_dti"] is None
    def network_down(request, timeout):
        raise OSError("offline")
    unavailable = lb.observe(network_down)
    assert unavailable["status"] == "unavailable"
    assert unavailable["error"] == "offline"
