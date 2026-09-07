# Dealer double-riichi yearly inference

- Source: `dealer-double-riichi-v2.0.0-yearly.json`
- Dataset: `NikkeTryHard/tenhou-to-mjai` `v2.0.0`
- Interval: two-sided 95% Wilson score interval
- Heterogeneity: Pearson chi-square without Yates correction

| Year | Dealer win | Dealer not win | Observations | Win rate | 95% Wilson CI | Difference from overall |
|---:|---:|---:|---:|---:|---:|---:|
| 2009 | 17 | 6 | 23 | 73.91% | [53.53%, 87.45%] | +3.61% |
| 2010 | 162 | 93 | 255 | 63.53% | [57.46%, 69.20%] | -6.77% |
| 2011 | 263 | 114 | 377 | 69.76% | [64.94%, 74.18%] | -0.54% |
| 2012 | 309 | 132 | 441 | 70.07% | [65.64%, 74.15%] | -0.23% |
| 2013 | 368 | 140 | 508 | 72.44% | [68.40%, 76.15%] | +2.14% |
| 2014 | 382 | 163 | 545 | 70.09% | [66.12%, 73.78%] | -0.21% |
| 2015 | 411 | 159 | 570 | 72.11% | [68.28%, 75.63%] | +1.80% |
| 2016 | 418 | 179 | 597 | 70.02% | [66.22%, 73.55%] | -0.29% |
| 2017 | 422 | 193 | 615 | 68.62% | [64.84%, 72.16%] | -1.68% |
| 2018 | 476 | 191 | 667 | 71.36% | [67.82%, 74.67%] | +1.06% |
| 2019 | 441 | 176 | 617 | 71.47% | [67.79%, 74.90%] | +1.17% |
| 2020 | 577 | 244 | 821 | 70.28% | [67.07%, 73.31%] | -0.02% |
| 2021 | 489 | 220 | 709 | 68.97% | [65.47%, 72.27%] | -1.33% |
| 2022 | 436 | 196 | 632 | 68.99% | [65.28%, 72.47%] | -1.31% |
| 2023 | 448 | 193 | 641 | 69.89% | [66.23%, 73.31%] | -0.41% |
| 2024 | 481 | 197 | 678 | 70.94% | [67.42%, 74.24%] | +0.64% |
| 2025 | 507 | 195 | 702 | 72.22% | [68.79%, 75.41%] | +1.92% |
| **Overall** | **6607** | **2791** | **9398** | **70.30%** | **[69.37%, 71.22%]** | **0.00%** |

## Heterogeneity

- Chi-square: 11.999520
- Degrees of freedom: 16
- p-value: 0.744013
- Cramér's V: 0.035733
- Minimum expected count: 6.830496
- Cells below 5: 0
- Cells below 1: 0
- Asymptotic conditions met: yes
- Zero-observation years excluded: none

## Interpretation limits

- Confidence-interval overlap is not used to decide statistical significance.
- The p-value does not measure the practical size of yearly differences; yearly deviations, the observed range, and Cramér's V must also be considered.
- A p-value above 0.05 means there is insufficient evidence to reject homogeneity, not that yearly differences do not exist.
- The aggregate input cannot evaluate dependence among rounds from the same game or player.
