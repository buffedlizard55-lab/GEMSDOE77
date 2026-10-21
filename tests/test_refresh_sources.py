"""Pure unit tests for the GitHub owner-mirror inventory discovery policy."""

from refresh_sources import FAMILY_REPOSITORY, _natural_family_key


def test_family_pattern_covers_historical_and_new_sibling_spellings_only():
    expected = ["GEMSDOE", "GEMSDOE2", "5GEMSDOE", "55GEMSDOE", "GEMSDOE79", "GEMSDOE82"]
    rejected = ["LEARNGEMSDOE", "GEMSDOE-79", "NOTGEMSDOE", "GEMSDOEfoo", "77-GEMSDOE"]
    assert all(FAMILY_REPOSITORY.fullmatch(name) for name in expected)
    assert all(not FAMILY_REPOSITORY.fullmatch(name) for name in rejected)


def test_family_sort_is_numeric_across_prefix_and_suffix_spellings():
    unsorted = [
        {"name": "GEMSDOE82", "default_branch": "main"},
        {"name": "55GEMSDOE", "default_branch": "main"},
        {"name": "GEMSDOE", "default_branch": "main"},
        {"name": "GEMSDOE2", "default_branch": "main"},
        {"name": "5GEMSDOE", "default_branch": "main"},
    ]
    assert [target["name"] for target in sorted(unsorted, key=_natural_family_key)] == [
        "GEMSDOE", "GEMSDOE2", "5GEMSDOE", "55GEMSDOE", "GEMSDOE82",
    ]
