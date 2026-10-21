"""Archive integrity measurements must never become submission authorization."""
import hashlib
import zipfile
import warnings
from pathlib import Path

import pytest

import audit_submission_archives as audit


def fixture(tmp_path, members):
    primary = tmp_path / "prediction.tif"
    primary.write_bytes(b"validated-primary-placeholder")
    archive = primary.with_suffix(".zip")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for name, data in members:
                dst.writestr(name, data)
    return archive, primary, hashlib.sha256(primary.read_bytes()).hexdigest()


def test_single_identical_member(tmp_path):
    args = fixture(tmp_path, [("prediction.tif", b"validated-primary-placeholder")])
    result = audit.validate_archive(*args)
    assert result["archive_pass"]
    assert result["authorized_to_submit"] is False


@pytest.mark.parametrize("members", [
    [], [("prediction.tif", b"wrong")],
    [("../prediction.tif", b"validated-primary-placeholder")],
    [("nested/prediction.tif", b"validated-primary-placeholder")],
    [("prediction.tif", b"validated-primary-placeholder"), ("extra.txt", b"x")],
    [("prediction.tif", b"validated-primary-placeholder"),
     ("prediction.tif", b"validated-primary-placeholder")],
    [("prediction.tif", b"X" * len(b"validated-primary-placeholder"))],
])
def test_bad_members_fail_closed(tmp_path, members):
    args = fixture(tmp_path, members)
    assert not audit.validate_archive(*args)["archive_pass"]


def test_bad_primary_pin(tmp_path):
    archive, primary, _ = fixture(tmp_path, [("prediction.tif", b"validated-primary-placeholder")])
    assert not audit.validate_archive(archive, primary, "0" * 64)["archive_pass"]


def test_corrupt_archive(tmp_path):
    args = fixture(tmp_path, [])
    args[0].write_bytes(b"not a zip")
    assert not audit.validate_archive(*args)["archive_pass"]


def test_symlink_member_rejected(tmp_path):
    archive, primary, sha = fixture(tmp_path, [])
    member = zipfile.ZipInfo(primary.name)
    member.create_system = 3
    member.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as dst:
        dst.writestr(member, primary.read_bytes())
    assert not audit.validate_archive(archive, primary, sha)["archive_pass"]


def test_all_current_archives_and_primary_formats():
    result = audit.audit_archives()
    assert result["all_archive_pass"]
    assert result["all_primary_format_pass"]
    assert result["checked_archive_count"] == len(result["artifacts"]) > 0
    assert not result["authorized_to_submit"]


def test_operative_status_precedes_historical_hero_cards():
    root = Path(__file__).resolve().parents[1]
    for name in ("index.html", "executive-summary.html"):
        text = (root / "docs" / name).read_text()
        main = text.split('<main><div class="wrap">', 1)[1]
        assert main.startswith('<section class="card" id="review-5999c8ab-status"')
        assert "Current review: NOT OK TO SUBMIT" in main[:180]
        assert "No new TIFF generated" in main[:250]
        assert "review-5999c8ab.html" in main[:1600]
