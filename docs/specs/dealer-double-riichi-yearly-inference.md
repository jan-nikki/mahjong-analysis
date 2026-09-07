# Dealer double-riichi yearly inference

## Scope and source

This analysis estimates uncertainty in the yearly dealer double-riichi win
rates and tests whether the rates are homogeneous across 2009 through 2025.
Its only input is the reviewed aggregate:

```text
research/results/dealer-double-riichi-v2.0.0-yearly.json
```

It does not read raw MJAI logs or reimplement any mahjong rule. For each year,
`x` is `dealer_win`, `n` is `dealer_double_riichi`, and
`dealer_not_win = n - x`. The latter includes both other-player wins and
draws.

The loader requires schema version 1, release tag `v2.0.0`, and exactly the
17 unique years 2009 through 2025 in ascending order. Count fields must be
non-negative integers, with booleans rejected as integers. For every year and
the total it requires:

```text
dealer_win + other_win + draw == dealer_double_riichi
```

Every total count must equal the sum of the yearly counts. Stored `win_rate`
values are never used as inputs; they are checked with `math.isclose` against
rates recalculated from integer counts.

## Wilson score interval

The confidence level is 95%, with `alpha = 0.05` and

```text
z = statistics.NormalDist().inv_cdf(0.975)
```

For `p_hat = x / n` and `d = 1 + z^2 / n`, the interval is:

```text
center = (p_hat + z^2 / (2n)) / d
half = z / d * sqrt(p_hat(1 - p_hat) / n + z^2 / (4n^2))
[lower, upper] = [max(0, center - half), min(1, center + half)]
```

When `n == 0`, the rate and interval are undefined and represented by JSON
`null` and Markdown `N/A`. When `x == 0`, the lower endpoint is fixed at
`0.0`; when `x == n`, the upper endpoint is fixed at `1.0`. Negative counts,
`x > n`, non-integers, and booleans raise `ValueError`.

These are separate 95% intervals for each year, not a simultaneous 95%
confidence region covering all 17 years.

## Pearson heterogeneity test

The observed table has one row per year and two columns:

```text
[dealer_win, dealer_not_win] = [x_i, n_i - x_i]
```

Years with `n_i == 0` contain no information and are excluded, with their
year identifiers reported. Let `k` be the number of included years,
`X = sum(x_i)`, `F = sum(n_i - x_i)`, and `N = X + F`. Expected counts are:

```text
E_i,win = n_i * X / N
E_i,not_win = n_i * F / N
```

The Pearson statistic, without Yates correction, is:

```text
chi_square = sum((observed - expected)^2 / expected)
```

The degrees of freedom are `(k - 1) * (2 - 1) = k - 1`. For the complete
2009–2025 input, `k == 17` and `df == 16`. The null hypothesis is that the
true dealer double-riichi win rate is identical in all included years. This
is an omnibus heterogeneity test; it is not a trend test and does not identify
which pairs of years differ.

The p-value is the chi-square survival probability
`P(ChiSquare(df) >= chi_square)`. No SciPy dependency is used. For even
degrees of freedom, the regularized upper incomplete gamma function is a
finite sum. For odd degrees of freedom, calculation starts from
`erfc(sqrt(chi_square / 2))` and uses the upper-gamma recurrence. Both paths
support every positive integer degree of freedom.

The test is not applicable when fewer than two non-empty years remain, or
when every observation is a win or every observation is a non-win. In these
cases the statistic, p-value, and Cramér's V are `null`, and a reason is
reported.

The output includes the minimum expected count and counts of cells below 5
and below 1. The asymptotic-condition flag follows the conventional rule that
no expected count is below 1 and no more than 20% are below 5. In the reviewed
dataset all 34 expected cells exceed 5; the smallest is the 2009 non-win cell,
approximately 6.83.

For an applicable 2-column table, effect size is Cramér's V:

```text
V = sqrt(chi_square / N)
```

## Output and interpretation

The default outputs are:

```text
outputs/dealer-double-riichi/yearly-inference.json
outputs/dealer-double-riichi/yearly-inference.md
```

JSON is the machine-readable source and Markdown is derived from the same
in-memory result. Both are deterministic and contain no timestamp, duration,
or absolute path. JSON retains unrounded floating-point values. Markdown
shows rates, intervals, and differences from the overall rate as percentages
with two decimal places; test statistics and Cramér's V use six decimal
places, and small p-values use scientific notation. Rounded display values
are never reused in calculations.

Confidence-interval overlap is not used as a significance test. The p-value
does not measure the practical size of a difference, so interpretation also
uses yearly differences from the overall rate, the observed rate range, and
Cramér's V. A p-value above 0.05 is described as insufficient evidence to
reject homogeneity, not as proof that yearly differences do not exist.

The aggregate input cannot identify dependence among rounds from the same
game or player. Wilson intervals and the Pearson test therefore use an
independent-observation approximation and do not account for game-level or
player-level clustering. Pairwise tests, multiple-comparison adjustments,
and time-trend models are outside this issue.

## Independent verification

Before promoting generated output into `research/results/`, compare Wilson
endpoints with `statsmodels.stats.proportion.proportion_confint(...,
method="wilson")` or R `prop.test(..., correct=FALSE)`, and compare the
contingency result with `scipy.stats.chi2_contingency(..., correction=False)`
or R `chisq.test(..., correct=FALSE)`. These tools are for external
verification only and are not project dependencies.
