from __future__ import annotations

import pytest

from analysis.pilot_combo_prediction import PILOT_FEATURE_SETS, parse_args


def test_pilot_feature_profile_remains_frozen() -> None:
    assert PILOT_FEATURE_SETS == (
        "simple",
        "tile_turn",
        "conventional",
        "conventional_simple",
        "conventional_simple_percentile",
    )


def test_cli_keeps_2025_out_of_development_choices() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--train-year", "2025"])

    with pytest.raises(SystemExit):
        parse_args(["--test-year", "2025"])


def test_cli_requires_out_of_time_year_order() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--train-year", "2022", "--test-year", "2021"])
