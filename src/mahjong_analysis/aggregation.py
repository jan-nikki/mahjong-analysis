"""Aggregation utilities for mahjong analysis."""

from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike

from mahjong_analysis.mjai import (
    classify_dealer_double_riichi_result,
    filter_east_kyokus,
    is_dealer_double_riichi,
    is_target_game,
    load_mjai,
    split_kyoku,
)


@dataclass(frozen=True)
class DealerDoubleRiichiStats:
    """Aggregated dealer double-riichi statistics for one year."""

    year: int
    target_games: int
    east_kyokus: int
    dealer_double_riichi: int
    dealer_win: int
    other_win: int
    draw: int

    @property
    def win_rate(self) -> float | None:
        """Return the dealer win rate, or None when no cases were observed."""
        if self.dealer_double_riichi == 0:
            return None
        return self.dealer_win / self.dealer_double_riichi


def aggregate_dealer_double_riichi(
    year: int,
    paths: Iterable[str | PathLike[str]],
) -> DealerDoubleRiichiStats:
    """Aggregate dealer double-riichi results from MJAI files."""
    target_games = 0
    east_kyokus = 0
    dealer_double_riichi = 0
    dealer_win = 0
    other_win = 0
    draw = 0

    for path in paths:
        try:
            events = load_mjai(path)

            if not is_target_game(path, events):
                continue

            target_games += 1
            kyokus = split_kyoku(events)
            east_kyokus_for_game = filter_east_kyokus(kyokus)
            east_kyokus += len(east_kyokus_for_game)

            for kyoku in east_kyokus_for_game:
                if not is_dealer_double_riichi(kyoku):
                    continue

                dealer_double_riichi += 1
                result = classify_dealer_double_riichi_result(kyoku)

                if result == "dealer_win":
                    dealer_win += 1
                elif result == "other_win":
                    other_win += 1
                elif result == "draw":
                    draw += 1
                else:
                    raise RuntimeError(
                        f"unexpected dealer double-riichi result: {result}"
                    )
        except Exception as error:
            raise RuntimeError(
                f"failed to process MJAI file: {path}"
            ) from error

    if dealer_win + other_win + draw != dealer_double_riichi:
        raise RuntimeError("dealer double-riichi result counts are inconsistent")

    return DealerDoubleRiichiStats(
        year=year,
        target_games=target_games,
        east_kyokus=east_kyokus,
        dealer_double_riichi=dealer_double_riichi,
        dealer_win=dealer_win,
        other_win=other_win,
        draw=draw,
    )
