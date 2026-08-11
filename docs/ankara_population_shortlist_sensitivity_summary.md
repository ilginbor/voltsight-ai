# Ankara Population Shortlist Sensitivity

## Purpose

This diagnostic compares the current canonical 20-site Ankara shortlist with
a shortlist ranked by the previously selected 5% residential-demand adjustment.

The canonical suitability files and canonical shortlist files are not modified.

## Fixed Selection Rules

Both scenarios use the same existing quality gates:

- Original suitability >= 60
- Feasibility >= 60
- Need >= 50
- Minimum representative-point spacing >= 25 km
- Desired shortlist size: 20

The current scenario orders eligible candidates by the original suitability
rank.

The adjusted scenario keeps the same original quality gates but orders eligible
candidates by:

`0.95 * current_suitability + 0.05 * balanced_population_demand`

This isolates ranking sensitivity from eligibility-policy changes.

## Shortlist Overlap

- Common selected cells: 15/20
- Overlap fraction: 75.00%
- Removed current cells: 5
- Added adjusted cells: 5

### Removed From Current Shortlist

- `ANK_007151`
- `ANK_010930`
- `ANK_028141`
- `ANK_075344`
- `ANK_080777`

### Added By 5% Demand Adjustment

- `ANK_007321`
- `ANK_028482`
- `ANK_080017`
- `ANK_080769`
- `ANK_097985`

## Scenario Metrics

| Metric | Current | 5% demand-adjusted |
| --- | ---: | ---: |
| Eligible candidates | 4,954 | 4,954 |
| Minimum spacing (km) | 25.080 | 25.045 |
| Median nearest-selected distance (km) | 27.106 | 27.230 |
| Mean pairwise distance (km) | 103.484 | 108.167 |
| Maximum pairwise distance (km) | 258.256 | 258.256 |
| Median original suitability | 73.4591 | 73.4591 |
| Minimum original suitability | 63.1078 | 62.9980 |
| Median feasibility | 74.8714 | 74.2100 |
| Minimum feasibility | 60.2973 | 60.0112 |
| Median need | 74.8971 | 75.5624 |
| Minimum need | 54.3497 | 54.3497 |
| Median demand | 84.8426 | 88.4174 |
| Minimum demand | 23.9295 | 23.9295 |
| Demand >= 70 | 17/20 | 18/20 |
| Worst original suitability rank | 8,728 | 8,804 |

## Interpretation Policy

This is a shortlist-level sensitivity test, not a new canonical scoring model.

A favorable 5% adjustment should increase residential-demand representation
without materially degrading original suitability, feasibility, infrastructure
need, or province-wide spatial spread.

Because the original quality thresholds are held fixed, any shortlist change
comes from ranking and the downstream greedy 25-km spacing interaction rather
than from relaxing candidate quality gates.

Population remains a modeled residential-demand proxy. It does not directly
measure traffic, employment, commuting, retail activity, tourism, EV ownership,
or electricity-grid capacity.

The final decision on whether to adopt a population adjustment should consider
this shortlist diagnostic together with the earlier ML incremental-value and
weight-sensitivity analyses.

## Outputs

- `data/processed/ankara_population_shortlist_sensitivity_current.csv`
- `data/processed/ankara_population_shortlist_sensitivity_adjusted.csv`
- `data/processed/ankara_population_shortlist_sensitivity_metrics.csv`
- `docs/ankara_population_shortlist_sensitivity.png`

## Generated At

2026-08-11T12:02:01.842297+00:00
