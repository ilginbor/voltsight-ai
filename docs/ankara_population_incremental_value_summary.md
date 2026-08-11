# Ankara Population Incremental Value

## Purpose

This experiment tests whether WorldPop-derived residential population adds
predictive ranking information beyond the deduplicated road-and-parking
baseline.

The experiment is incremental rather than a new tuned model search.

## Dataset

- Rows: 102,745
- Positive existing-station cells: 46
- Spatial folds: 5
- Spatial block size: 5 km
- Population source: WorldPop 2025 constrained R2024B

## Feature Sets

### Normalized 12

The previously audited deduplicated road-and-parking baseline. It excludes
`road_length_m` and `parking_area_m2` while retaining their normalized
counterparts.

### Normalized 12 + Local Population

Adds `population_count` only.

`population_density_per_km2` is intentionally excluded because every analysis
cell is 0.25 km2, making density a deterministic scale transform of local
population count.

### Normalized 12 + Population Context

Adds:

- `population_count`
- `population_within_1000m`
- `population_within_2000m`

The neighborhood variables test whether surrounding residential demand adds
information beyond the local 500-m cell.

## Models

The existing untuned Logistic Regression, Random Forest, and
HistGradientBoosting configurations are reused unchanged. The same predefined
5-km spatial folds and the same class-imbalance treatments are retained. No
hyperparameter search is performed.

## Spatial OOF Results

| Model | Feature set | Features | Pooled AP | Delta AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Delta top 1% | Top 5% recall | Delta top 5% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Normalized 12 | 12 | 0.031057 | +0.000000 | 0.050472 | 0.024529 | 0.967787 | 0.543478 | +0.000000 | 0.891304 | +0.000000 |
| Logistic Regression | Normalized 12 + local population | 13 | 0.032312 | +0.001255 | 0.054556 | 0.023701 | 0.967104 | 0.543478 | +0.000000 | 0.891304 | +0.000000 |
| Logistic Regression | Normalized 12 + local + 1 km + 2 km population | 15 | 0.031069 | +0.000011 | 0.048221 | 0.025643 | 0.965129 | 0.500000 | -0.043478 | 0.891304 | +0.000000 |
| Random Forest | Normalized 12 | 12 | 0.083300 | +0.000000 | 0.110030 | 0.126112 | 0.948239 | 0.434783 | +0.000000 | 0.869565 | +0.000000 |
| Random Forest | Normalized 12 + local population | 13 | 0.074946 | -0.008353 | 0.100237 | 0.128181 | 0.958484 | 0.413043 | -0.021739 | 0.913043 | +0.043478 |
| Random Forest | Normalized 12 + local + 1 km + 2 km population | 15 | 0.089388 | +0.006088 | 0.093487 | 0.087163 | 0.941260 | 0.391304 | -0.043478 | 0.891304 | +0.021739 |
| HistGradientBoosting | Normalized 12 | 12 | 0.063939 | +0.000000 | 0.091578 | 0.078839 | 0.938261 | 0.478261 | +0.000000 | 0.826087 | +0.000000 |
| HistGradientBoosting | Normalized 12 + local population | 13 | 0.059771 | -0.004168 | 0.100726 | 0.111842 | 0.932512 | 0.478261 | +0.000000 | 0.782609 | -0.043478 |
| HistGradientBoosting | Normalized 12 + local + 1 km + 2 km population | 15 | 0.053085 | -0.010854 | 0.079871 | 0.077136 | 0.970690 | 0.521739 | +0.043478 | 0.869565 | +0.043478 |

## Full Population-Context Delta Against Normalized 12

- Logistic Regression: pooled AP delta +0.000011, top-1% recall delta -0.043478, top-5% recall delta +0.000000.
- Random Forest: pooled AP delta +0.006088, top-1% recall delta -0.043478, top-5% recall delta +0.021739.
- HistGradientBoosting: pooled AP delta -0.010854, top-1% recall delta +0.043478, top-5% recall delta +0.043478.

## Interpretation Policy

Average precision is primary because only a very small fraction of Ankara grid
cells contain known existing charging stations. Top-1% and top-5% recall are
also reported because VoltSight is a candidate-ranking system.

A positive delta means the population feature set improved spatial OOF ranking
under this experiment. It is predictive evidence, not a causal estimate of the
real-world effect of population on station placement.

Population represents modeled residential demand only. It does not directly
capture employment, commuting, retail activity, tourism, traffic volume,
vehicle ownership, or electricity-grid capacity.

Only 46 positive station cells are available, so fold-level variability must
be considered alongside pooled metrics. The existing spatial block design
reduces local dependence but does not eliminate all spatial autocorrelation.

The historical full-14 baselines remain historical references. This experiment
uses normalized-12 as the deduplicated baseline for future feature-family
evaluation.

## Outputs

- `data/processed/ankara_population_incremental_value_metrics.csv`
- `data/processed/ankara_population_incremental_value_fold_metrics.csv`
- `data/processed/ankara_population_incremental_value_oof_predictions.csv`
- `docs/ankara_population_incremental_value.png`

## Generated At

2026-08-11T11:33:05.595738+00:00
