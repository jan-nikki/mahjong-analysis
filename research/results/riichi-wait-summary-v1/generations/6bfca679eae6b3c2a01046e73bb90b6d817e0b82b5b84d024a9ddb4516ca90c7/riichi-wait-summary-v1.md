# Established-riichi wait baseline

- Input: `data/processed/riichi-waits-v1/manifest.json`
- Records: 10,706,714
- Observation unit: one established-riichi record
- Headline ryanmen metric: `is_pure_ryanmen`

## Overall

| Metric | Formal | Self-ownership adjusted |
|---|---:|---:|
| Pure ryanmen | 41.72% | 41.71% |
| Contains ryanmen | 59.82% | 59.82% |
| Multiwait | 8.22% | 8.22% |

Wait-shape and hand-type memberships are non-exclusive; their rates need not sum to 100%.

## Yearly headline

| Year | Records | Pure ryanmen | Contains ryanmen | Multiwait |
|---:|---:|---:|---:|---:|
| 2009 | 28,298 | 41.79% | 59.94% | 8.26% |
| 2010 | 284,903 | 41.53% | 59.74% | 8.32% |
| 2011 | 401,355 | 41.45% | 59.67% | 8.25% |
| 2012 | 458,210 | 41.32% | 59.34% | 8.26% |
| 2013 | 520,642 | 41.18% | 59.09% | 8.22% |
| 2014 | 573,214 | 41.48% | 59.64% | 8.34% |
| 2015 | 625,442 | 41.36% | 59.38% | 8.18% |
| 2016 | 665,852 | 41.89% | 60.00% | 8.23% |
| 2017 | 716,305 | 42.32% | 60.67% | 8.37% |
| 2018 | 744,797 | 41.89% | 60.21% | 8.37% |
| 2019 | 751,499 | 41.63% | 59.79% | 8.28% |
| 2020 | 975,174 | 41.47% | 59.46% | 8.15% |
| 2021 | 831,676 | 41.57% | 59.51% | 8.15% |
| 2022 | 751,967 | 41.66% | 59.64% | 8.13% |
| 2023 | 756,643 | 41.81% | 59.81% | 8.06% |
| 2024 | 825,963 | 42.03% | 60.13% | 8.15% |
| 2025 | 794,774 | 42.27% | 60.49% | 8.19% |

## Exact riichi discard number

| Discard number | Records | Pure ryanmen | Contains ryanmen | Multiwait |
|---:|---:|---:|---:|---:|
| 1 | 36,213 | 20.51% | 30.35% | 4.02% |
| 2 | 85,877 | 32.35% | 46.84% | 6.05% |
| 3 | 225,316 | 34.90% | 50.71% | 6.58% |
| 4 | 452,592 | 36.55% | 53.35% | 7.18% |
| 5 | 743,602 | 38.22% | 55.46% | 7.46% |
| 6 | 1,036,969 | 39.56% | 57.26% | 7.75% |
| 7 | 1,258,554 | 40.70% | 58.70% | 8.01% |
| 8 | 1,356,865 | 41.95% | 60.15% | 8.23% |
| 9 | 1,318,221 | 42.98% | 61.30% | 8.38% |
| 10 | 1,174,038 | 43.75% | 62.23% | 8.51% |
| 11 | 969,498 | 44.36% | 62.96% | 8.64% |
| 12 | 749,155 | 44.53% | 63.19% | 8.69% |
| 13 | 537,908 | 44.44% | 63.22% | 8.85% |
| 14 | 362,164 | 43.88% | 62.76% | 9.01% |
| 15 | 224,265 | 42.85% | 62.01% | 9.25% |
| 16 | 123,060 | 42.10% | 61.48% | 9.80% |
| 17 | 46,935 | 41.52% | 60.83% | 10.11% |
| 18 | 5,305 | 39.26% | 58.06% | 9.37% |
| 19 | 176 | 42.05% | 55.68% | 5.68% |
| 20 | 1 | 0.00% | 0.00% | 0.00% |

## Fifth-tile sensitivity

- Affected records: 305 (0.00%)
- Adjusted zero-wait records: 1
- The adjusted series removes only self-owned-four formal wait tiles; it is not a live-wait estimate.

## Validation

- Manifest SHA256: `2bbe3409fb8d8ba4e5c34f7531592e3f7e0a1685215f2617f91e27cbdbaa7127`
- Analysis commit: `7d19411ff46136c511a98af92e533f5ff1c5b3ff`
- Annual file integrity, DTO validation, annual counts, total counts, and aggregation invariants passed before these documents were built.
