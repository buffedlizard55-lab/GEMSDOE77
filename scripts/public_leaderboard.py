"""Read a *public participant row* only; never infer an artifact's organizer score.

The public DrivenData table does not expose submission filenames or SHA-256s.
This best-effort scheduled observation must fail closed if the page is blocked,
redirected to login, malformed, or changes markup. No auth, model, or submission.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import urllib.request
from html.parser import HTMLParser

URL = "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/"
MAX_BYTES = 2_000_000


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.depth = 0
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.depth == 0:
                self.table = []
            self.depth += 1
        elif self.depth == 1 and tag == "tr":
            self.row = []
        elif self.depth == 1 and tag in {"th", "td"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.depth == 1:
            if self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.depth:
            self.depth -= 1
            if self.depth == 0:
                self.tables.append(self.table)
                self.table = None


def first_public_row(document: str) -> dict:
    tables = Tables()
    tables.feed(document)
    for table in tables.tables:
        if not table or "rank" not in table[0][0].lower():
            continue
        score_cols = [i for i, name in enumerate(table[0])
                      if "public" in name.lower() and "tversky" in name.lower()]
        if len(score_cols) != 1:
            continue
        col = score_cols[0]
        leaders = [r for r in table[1:] if len(r) > col and r[0].strip() == "#1"]
        if len(leaders) != 1 or len(leaders[0]) < 3:
            raise ValueError("public leaderboard does not have exactly one complete #1 row")
        cells = leaders[0]
        score = cells[col].strip()
        if not re.fullmatch(r"(?:0|1)\.\d{4}", score) or not 0 <= float(score) <= 1:
            raise ValueError("top public DTI column is not a four-decimal [0,1] value")
        return {"rank": 1, "participant_row": cells[2][:180], "public_dti": float(score)}
    raise ValueError("official public leaderboard table/header not identified")


def observe(opener=urllib.request.urlopen, now=None) -> dict:
    checked = (now or dt.datetime.now(dt.timezone.utc)).isoformat()
    result = {"evidence_class": "PUBLIC-LEADERBOARD OBSERVATION (not ORGANIZER-CONFIRMED)",
              "url": URL, "checked_utc": checked, "status": "unavailable",
              "rank": None, "participant_row": None, "public_dti": None,
              "submission_name": None, "raster_sha256": None,
              "score_is_submission_page_receipt": False,
              "interpretation": "A public participant row never identifies an exact GeoTIFF or authorizes a slot."}
    request = urllib.request.Request(URL, headers={
        "User-Agent": "GEMSDOE77-source-health/1.0 (public research inventory)",
        "Accept": "text/html",
    })
    try:
        with opener(request, timeout=15) as response:
            actual = response.geturl()
            if actual != URL:
                raise ValueError("redirect or unexpected final URL (possibly login)")
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("response is not HTML")
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("leaderboard response exceeds size limit")
        row = first_public_row(raw.decode("utf-8", errors="strict"))
        result.update(row, status="retrieved", html_sha256=hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError, UnicodeError) as exc:
        result["error"] = str(exc)[:300]
    return result
