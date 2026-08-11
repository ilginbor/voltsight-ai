# Ankara Candidate ML Support Diagnostic

## Purpose

This diagnostic compares the canonical explainable suitability ranking with
Canonical Activity15 machine-learning evidence without blending the two into a
single decision score.

The ML side uses fold-normalized spatial OOF ranks rather than globally ranking
raw OOF scores from different fold-specific estimators.

Suitability remains the primary site-selection layer.

## Why OOF Scores Are Used

The canonical candidate set is exactly the target-negative subset of the
existing-station ML universe.

Each model score used here is therefore the spatial out-of-fold score generated
for that grid cell while its 5-km spatial fold was held out.

No full-data model is fitted to generate the candidate-support values in this
diagnostic.

This reduces in-sample optimism, but it is still internal spatial validation and
not independent external or temporal validation.

## Candidate ML Percentiles

Raw model scores come from different fold-specific estimators, so their absolute
scales are not assumed to be directly comparable across folds.

For each model, scores are converted to 0-100 candidate percentiles **within the
held-out spatial fold that produced them**. This fold-normalized ranking reduces
artifacts from fold-specific score scale or calibration differences.

Higher percentile means stronger within-fold agreement with the historical
spatial pattern learned by that model.

The cross-model ML consensus is the median of:

- Logistic Regression candidate percentile
- Random Forest candidate percentile
- HistGradientBoosting candidate percentile

The cross-model median is therefore a consensus of fold-normalized ranking
positions. Percentiles are ranking diagnostics, not calibrated probabilities.

## Province-Wide Agreement

- Candidate count: 102,699
- Spearman correlation, suitability vs ML consensus: 0.6865
- Top 1% overlap: 12/1,027 (1.17%)
- Top 5% overlap: 456/5,135 (8.88%)
- Top 10% overlap: 3,123/10,270 (30.41%)

## Canonical 20-Site Shortlist

- Median ML consensus percentile: 95.03
- Minimum ML consensus percentile: 83.66
- Maximum ML consensus percentile: 99.14
- Median cross-model percentile spread: 3.00
- At least two models in candidate top 20%: 20/20
- All three models in candidate top 20%: 15/20
- At least two models in candidate top 10%: 19/20

| # | Grid | Suitability rank | Suitability | LR pct | RF pct | HGB pct | ML consensus | Min model pct | Spread | Models top 20% |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `ANK_004300` | 1 | 89.7487 | 72.11 | 98.69 | 92.65 | 92.65 | 72.11 | 26.58 | 2/3 |
| 2 | `ANK_055975` | 17 | 84.8524 | 80.14 | 98.83 | 99.77 | 98.83 | 80.14 | 19.63 | 3/3 |
| 3 | `ANK_007151` | 21 | 84.1993 | 96.05 | 92.87 | 92.26 | 92.87 | 92.26 | 3.79 | 3/3 |
| 4 | `ANK_013005` | 31 | 83.3316 | 98.90 | 98.81 | 99.89 | 98.90 | 98.81 | 1.09 | 3/3 |
| 5 | `ANK_101065` | 102 | 81.4958 | 94.31 | 94.96 | 96.35 | 94.96 | 94.31 | 2.04 | 3/3 |
| 6 | `ANK_030609` | 442 | 78.6712 | 99.70 | 97.19 | 98.38 | 98.38 | 97.19 | 2.51 | 3/3 |
| 7 | `ANK_037238` | 554 | 78.0298 | 96.73 | 94.71 | 93.72 | 94.71 | 93.72 | 3.00 | 3/3 |
| 8 | `ANK_044917` | 903 | 76.6984 | 98.21 | 99.14 | 99.74 | 99.14 | 98.21 | 1.53 | 3/3 |
| 9 | `ANK_093363` | 2,007 | 73.7051 | 96.64 | 96.51 | 97.87 | 96.64 | 96.51 | 1.35 | 3/3 |
| 10 | `ANK_066425` | 2,086 | 73.5407 | 31.07 | 97.43 | 97.66 | 97.43 | 31.07 | 66.59 | 2/3 |
| 11 | `ANK_102631` | 2,165 | 73.3776 | 98.63 | 96.57 | 98.35 | 98.35 | 96.57 | 2.05 | 3/3 |
| 12 | `ANK_057005` | 2,690 | 72.2018 | 98.10 | 98.70 | 96.88 | 98.10 | 96.88 | 1.83 | 3/3 |
| 13 | `ANK_061180` | 3,284 | 71.0410 | 91.35 | 94.79 | 99.36 | 94.79 | 91.35 | 8.01 | 3/3 |
| 14 | `ANK_010010` | 3,697 | 70.3241 | 76.80 | 94.12 | 95.19 | 94.12 | 76.80 | 18.38 | 2/3 |
| 15 | `ANK_073387` | 4,936 | 68.3961 | 95.09 | 95.28 | 2.94 | 95.09 | 2.94 | 92.34 | 2/3 |
| 16 | `ANK_075344` | 5,428 | 67.6843 | 96.98 | 97.69 | 96.89 | 96.98 | 96.89 | 0.80 | 3/3 |
| 17 | `ANK_093670` | 5,614 | 67.4320 | 83.66 | 46.08 | 91.94 | 83.66 | 46.08 | 45.86 | 2/3 |
| 18 | `ANK_080777` | 6,193 | 66.6460 | 97.18 | 93.89 | 93.52 | 93.89 | 93.52 | 3.67 | 3/3 |
| 19 | `ANK_028141` | 6,415 | 66.3131 | 93.60 | 94.12 | 91.99 | 93.60 | 91.99 | 2.12 | 3/3 |
| 20 | `ANK_010930` | 8,728 | 63.1078 | 93.22 | 94.12 | 91.13 | 93.22 | 91.13 | 2.99 | 3/3 |

## Strongest ML-Supported Shortlist Cells

- `ANK_044917`: suitability rank 903, ML consensus 99.14
- `ANK_013005`: suitability rank 31, ML consensus 98.90
- `ANK_055975`: suitability rank 17, ML consensus 98.83
- `ANK_030609`: suitability rank 442, ML consensus 98.38
- `ANK_102631`: suitability rank 2,165, ML consensus 98.35

## Lowest ML-Consensus Shortlist Cells

- `ANK_093670`: suitability rank 5,614, ML consensus 83.66
- `ANK_004300`: suitability rank 1, ML consensus 92.65
- `ANK_007151`: suitability rank 21, ML consensus 92.87
- `ANK_010930`: suitability rank 8,728, ML consensus 93.22
- `ANK_028141`: suitability rank 6,415, ML consensus 93.60

A lower ML percentile does not invalidate a suitability recommendation.

Suitability explicitly rewards infrastructure need and feasibility, while the
ML models learn patterns associated with the limited existing-station
distribution. A high-suitability / lower-ML-agreement candidate may therefore
represent a gap-oriented recommendation that differs from historical placement
patterns.

## Interpretation Policy

This diagnostic must not be used as a new blended canonical score.

Fold normalization makes the OOF ranking scales more comparable, but it also
means each candidate is interpreted relative to other candidates in the same
held-out spatial fold before cross-model consensus is formed.

It provides a second axis of evidence:

- suitability: explainable forward-looking decision support
- ML percentile: agreement with historical mapped station-placement patterns

The ML signal is limited by only 46 positive station cells, incomplete open-data
coverage, residual spatial dependence, and the absence of independent external
validation.

Mapped OSM activity is a proxy rather than direct EV demand, traffic, trips,
employment, or commercial turnover.

## Outputs

- `data/processed/ankara_candidate_ml_support.csv`
- `data/processed/ankara_shortlist_ml_support.csv`
- `data/processed/ankara_candidate_ml_support_metrics.csv`
- `docs/ankara_suitability_ml_support.png`

## Generated At

2026-08-11T14:55:47.114547+00:00
