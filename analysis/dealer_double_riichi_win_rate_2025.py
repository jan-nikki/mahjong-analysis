"""Aggregate dealer double-riichi results for 2025."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from mahjong_analysis.aggregation import (
    DealerDoubleRiichiStats,
    aggregate_dealer_double_riichi,
)


YEAR = 2025
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / str(YEAR)
PROGRESS_INTERVAL = 10_000


def with_progress(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield paths and report progress after each completed interval."""
    for index, path in enumerate(paths, 1):
        yield path
        if index % PROGRESS_INTERVAL == 0:
            print(f"processed: {index:,} files", flush=True)


def print_stats(stats: DealerDoubleRiichiStats) -> None:
    """Print aggregate statistics to the console."""
    win_rate = "N/A" if stats.win_rate is None else f"{stats.win_rate:.2%}"

    print(f"year: {stats.year}")
    print(f"target_games: {stats.target_games}")
    print(f"east_kyokus: {stats.east_kyokus}")
    print(f"dealer_double_riichi: {stats.dealer_double_riichi}")
    print(f"dealer_win: {stats.dealer_win}")
    print(f"other_win: {stats.other_win}")
    print(f"draw: {stats.draw}")
    print(f"win_rate: {win_rate}")


def main() -> None:
    """Aggregate and print dealer double-riichi results for 2025."""
    paths = sorted(DATA_DIR.glob("*.mjson"), key=lambda path: path.name)
    stats = aggregate_dealer_double_riichi(YEAR, with_progress(paths))
    print_stats(stats)


if __name__ == "__main__":
    main()
