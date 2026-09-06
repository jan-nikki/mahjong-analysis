# Dataset validation specification

## Scope

This validation fixes and records the analysis dataset derived from
`NikkeTryHard/tenhou-to-mjai` release `v2.0.0`, covering the selected years
from 2009 through 2025.

The target-game condition is exactly:

- the filename rule code is `00a9`; and
- the first `start_game` event has `aka_flag is True`.

`aka_flag` that is missing or whose type is not exactly `bool` is excluded
from target games and recorded as a validation issue. In particular, integer
values `0` and `1` are invalid rather than `False` and `True`.

## Release asset manifest

`data/validation/tenhou-to-mjai-v2.0.0-assets.json` contains only values
obtained from GitHub release asset metadata:

- repository and release tag;
- asset year and ZIP filename;
- expected ZIP size in bytes; and
- expected SHA-256 digest.

ZIP entry counts and extracted-file counts are local observations. They are
not independent expected values in the asset manifest.

For each selected year, validation compares the expected size and SHA-256
with the local ZIP. SHA-256 is calculated incrementally rather than loading
the complete archive into memory. After that, it compares both the count and
the filename set of non-directory `.mjson` ZIP entries with direct-child
`.mjson` files under `data/raw/YYYY/`.

The ZIP and raw filename sets are retained only while their year is being
compared. The summary records whether they match, archive-only and raw-only
counts, and up to ten lexicographically sorted examples from each side. It
does not store a complete filename list. If the ZIP cannot be read, filename
comparison is unavailable rather than generating one issue per raw file.

The extraction-identity checks therefore proceed as:

1. verify the ZIP SHA-256 against the release manifest;
2. collect the ZIP's non-directory `.mjson` filename set; and
3. compare it with the direct-child raw `.mjson` filename set.

## Raw MJAI inspection

Only direct-child files matching `data/raw/YYYY/*.mjson` are inspected.
Other files and subdirectories directly below the yearly directory are
recorded as unexpected entries and validation issues.

For each `.mjson`, validation performs only these operations:

1. validate the filename with the production filename specification;
2. open the file in binary mode and inspect its first two bytes;
3. classify it as gzip when the bytes are `1f 8b`, otherwise as plain;
4. read only the first decompressed line;
5. decode that line as UTF-8 and parse it as JSON; and
6. inspect `type` and `aka_flag` on the first event.

Compression is observed from the magic bytes. It is never inferred from the
year, extension, or directory name. A binary-open failure has compression
`unknown`.

The first-event states distinguish a readable `start_game`, an empty file,
JSON decoding failure, a non-object JSON value, a different event type, and
an unreadable first line. The `aka_flag` states are `true`, `false`,
`missing`, `invalid`, and `unavailable`. `missing` applies only to a readable
`start_game`; when the first event cannot be obtained and validated, the
state is `unavailable`.

An invalid filename does not prevent first-event inspection. Its rule code
and filename year are unavailable. A valid filename whose year differs from
its containing yearly directory is a validation issue.

## Errors and output

File-level errors do not stop the yearly scan. Counts are kept by category,
while representative relative paths are retained in deterministic filename
order up to ten per category. Absolute paths are not written. The count of
`.mjson` files with at least one issue is kept separately from issue-category
counts because one file can have multiple independent issues.

The canonical machine-readable output is:

`data/validation/tenhou-to-mjai-v2.0.0-summary.json`

The deterministic human-readable report derived from the same in-memory
summary is:

`data/validation/tenhou-to-mjai-v2.0.0-summary.md`

Both outputs omit execution timestamps and absolute paths. They are written
through temporary files and then replaced. Data validation failures are
written to both outputs before the command exits with a nonzero status.

## Command-line execution

`analysis/validate_dataset_v2_0_0.py` validates all years from 2009 through
2025 by default. `--years` selects one or more years in that range for a
shorter development run. Files are processed sequentially, one yearly
directory at a time, with progress reported every 10,000 `.mjson` files.

`--skip-archive-hash` omits archive SHA-256 calculation for development. A
canonical summary must be generated without that option. A partial-year run
records its selected years in the output; a separate output path should be
used when the canonical all-year summary must be preserved.

ZIP entries not ending in `.mjson` are recorded as observed unexpected
entries. A mismatch between the observed ZIP `.mjson` count and the observed
raw `.mjson` count is a validation failure. A ZIP/raw `.mjson` filename-set
mismatch is a separate validation failure, including when the counts happen
to be equal. Each yearly mismatch produces one aggregate issue rather than
one issue for every missing or extra filename.

## Limitations

This is a fast dataset-identity and first-event structure check. It does not
read or validate every MJAI event, detect malformed JSON after the first
line, or verify a gzip trailer that is not encountered while reading the
first line.

A matching ZIP SHA-256 confirms that the distributed archive itself is the
release asset identified by the manifest. Matching ZIP-entry and raw-file
counts confirms the number of extracted logs, while matching filename sets
confirms which named logs were extracted. This validation does not compare
the contents of individual extracted raw files byte-for-byte with their
corresponding ZIP entries.
