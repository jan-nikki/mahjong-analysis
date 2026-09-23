from __future__ import annotations

from pathlib import Path

import pytest

from mahjong_analysis.combo_dataset import (
    DEVELOPMENT_YEARS,
    expanding_year_splits,
    select_stable_year_sample,
)


def _touch_sources(root: Path, year: int, names: list[str]) -> None:
    year_root = root / str(year)
    year_root.mkdir(parents=True)
    for name in names:
        (year_root / name).touch()


def test_stable_sample_is_deterministic_and_hash_ordered(tmp_path: Path) -> None:
    _touch_sources(
        tmp_path,
        2020,
        ["z.mjson", "a.mjson", "m.mjson", "ignored.txt", "b.mjson"],
    )

    first = select_stable_year_sample(tmp_path, 2020, 3, seed=17)
    second = select_stable_year_sample(tmp_path, 2020, 3, seed=17)

    assert first == second
    assert first.candidate_count == 4
    assert len(first.paths) == 3
    assert first.selection_digests == tuple(sorted(first.selection_digests))
    assert all(source.startswith("2020/") for source in first.relative_sources)


def test_stable_sample_seed_changes_selection_score(tmp_path: Path) -> None:
    _touch_sources(tmp_path, 2020, [f"{index}.mjson" for index in range(10)])

    first = select_stable_year_sample(tmp_path, 2020, 4, seed=1)
    second = select_stable_year_sample(tmp_path, 2020, 4, seed=2)

    assert first.selection_digests != second.selection_digests
    assert first.relative_sources != second.relative_sources


def test_stable_sample_rejects_reserved_years_and_oversampling(
    tmp_path: Path,
) -> None:
    _touch_sources(tmp_path, 2020, ["only.mjson"])

    with pytest.raises(ValueError, match="requested 2"):
        select_stable_year_sample(tmp_path, 2020, 2, seed=1)
    with pytest.raises(ValueError, match="outside development years"):
        select_stable_year_sample(tmp_path, 2024, 1, seed=1)
    with pytest.raises(ValueError, match="outside development years"):
        select_stable_year_sample(tmp_path, 2025, 1, seed=1)


def test_expanding_year_splits_are_strictly_out_of_time() -> None:
    assert DEVELOPMENT_YEARS == (2020, 2021, 2022, 2023)
    assert expanding_year_splits() == (
        ((2020,), 2021),
        ((2020, 2021), 2022),
        ((2020, 2021, 2022), 2023),
    )


@pytest.mark.parametrize(
    "years",
    [
        (2020,),
        (2021, 2020),
        (2020, 2020),
    ],
)
def test_expanding_year_splits_reject_invalid_sequences(years: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        expanding_year_splits(years)
