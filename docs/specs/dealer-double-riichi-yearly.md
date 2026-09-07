# Dealer double-riichi yearly aggregation

## Scope

This analysis uses the `NikkeTryHard/tenhou-to-mjai` `v2.0.0` dataset and
supports years 2009 through 2025. A run must explicitly select one or more
years with `--years`, or every supported year with `--all`. The two options
are mutually exclusive and omitting both is an error.

Target games have filename rule code `00a9` and a first `start_game` event
whose `aka_flag is True`. Only East-round kyokus whose `bakaze == "E"` are
included. Established dealer double-riichi and its result are determined by
the production functions in `mahjong_analysis.mjai`; the yearly layer does
not reproduce mahjong rules.

## Dataset preconditions

Before reading complete MJAI logs, the command loads
`data/validation/tenhou-to-mjai-v2.0.0-summary.json` and requires:

- release tag `v2.0.0`;
- `validation_failed == false`;
- `archive_hashes_verified == true`; and
- a raw-file count for every selected year.

Each `data/raw/YYYY` directory must exist. Its direct-child `.mjson` files
are sorted by filename and their count must match the validation summary
before the year's aggregation begins. `target_games` is not copied from the
dataset summary; it is independently calculated from the complete MJAI logs.

## Processing and errors

Years are unique and processed in ascending order. The current year's path
list is explicitly released before the next year is enumerated, and only the
current file's loaded events are retained by the production aggregator.
Processing is sequential, with progress reported every 10,000 completed files
as:

```text
2020: processed 10,000 files
```

Invalid MJAI is not skipped. The yearly layer adds the year to the exception
chain, while the production aggregator identifies the failing path. No
partial or resume output is written.

Independently verified integer results are stored in
`data/validation/dealer-double-riichi-v2.0.0-known-results.json`. When a
selected year has a known result, every stored integer count must match after
aggregation. `win_rate` is computed from counts and is not independently
compared.

## Output

Canonical machine-readable JSON and derived Markdown are written only after
all selected years complete successfully. The two output paths must remain
different after path resolution. Defaults are:

- `outputs/dealer-double-riichi/yearly.json`
- `outputs/dealer-double-riichi/yearly.md`

Both paths can be overridden. Outputs contain a schema version, dataset
repository and release tag, structured analysis conditions, selected years,
yearly results, and totals. They do not contain timestamps, durations, or
absolute paths. No CSV is generated.

Both complete UTF-8 temporary files are created before canonical replacement
begins. Each individual file is replaced atomically. The pair is not claimed
to be atomic across power loss or every OS failure. If the second replacement
fails, a best-effort rollback restores both paths to their prior contents or
absence. If rollback also fails, the original write error remains the cause of
an explicit error stating that output consistency could not be guaranteed.
Temporary and rollback files are removed after normal completion.

For every year and for totals:

```text
dealer_win + other_win + draw == dealer_double_riichi
```

The win rate is `dealer_win / dealer_double_riichi`. It is JSON `null` and
Markdown `N/A` when there are no observations. The total win rate is computed
from total wins and total observations, not by averaging yearly rates.

Generated files under `outputs/` are ignored by Git. Promotion of reviewed
full-period results into a tracked research artifact is a separate manual
decision after full validation.
