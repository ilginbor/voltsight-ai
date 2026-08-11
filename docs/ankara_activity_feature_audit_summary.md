# Ankara Activity Feature Audit

## Purpose

This audit evaluates coverage, sparsity, redundancy, and descriptive association
with known charging-station cells before OSM activity features are used in any
machine-learning or suitability model.

## Dataset

- Grid/training rows: 102,745
- Positive existing-station cells: 46
- Negative cells: 102,699
- Activity features: 11

## Coverage and Distribution

| Feature | Nonzero cells | Nonzero % | Median | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| poi_count | 3,026 | 2.95% | 0.00 | 0.00 | 0.00 | 5.00 | 303 |
| retail_commercial_count | 1,451 | 1.41% | 0.00 | 0.00 | 0.00 | 1.00 | 275 |
| education_count | 989 | 0.96% | 0.00 | 0.00 | 0.00 | 0.00 | 12 |
| healthcare_count | 866 | 0.84% | 0.00 | 0.00 | 0.00 | 0.00 | 15 |
| transport_activity_count | 2,263 | 2.20% | 0.00 | 0.00 | 0.00 | 2.00 | 49 |
| poi_count_within_1000m | 9,580 | 9.32% | 0.00 | 0.00 | 4.00 | 79.00 | 715 |
| poi_count_within_2000m | 19,258 | 18.74% | 0.00 | 5.00 | 19.00 | 324.00 | 1565 |
| retail_commercial_within_1000m | 4,499 | 4.38% | 0.00 | 0.00 | 0.00 | 34.00 | 508 |
| education_within_1000m | 3,824 | 3.72% | 0.00 | 0.00 | 0.00 | 7.00 | 31 |
| healthcare_within_1000m | 2,910 | 2.83% | 0.00 | 0.00 | 0.00 | 9.00 | 79 |
| transport_activity_within_1000m | 7,767 | 7.56% | 0.00 | 0.00 | 3.00 | 28.00 | 210 |

Sparse zero-heavy local counts are expected because OSM activity mapping is
concentrated in settlements and because the study area covers the full Ankara
province.

## Existing-Station Descriptive Comparison

| Feature | Positive median | Negative median | Positive nonzero | Negative nonzero | SMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| transport_activity_within_1000m | 27.50 | 0.00 | 89.13% | 7.52% | +1.5993 |
| poi_count_within_2000m | 389.50 | 0.00 | 100.00% | 18.71% | +1.5713 |
| education_within_1000m | 4.00 | 0.00 | 82.61% | 3.69% | +1.5173 |
| poi_count_within_1000m | 92.50 | 0.00 | 97.83% | 9.28% | +1.4351 |
| poi_count | 9.50 | 0.00 | 91.30% | 2.91% | +1.2215 |
| retail_commercial_within_1000m | 51.50 | 0.00 | 95.65% | 4.34% | +1.1694 |
| healthcare_within_1000m | 8.00 | 0.00 | 84.78% | 2.80% | +1.1155 |
| transport_activity_count | 2.00 | 0.00 | 65.22% | 2.17% | +1.1029 |
| retail_commercial_count | 5.00 | 0.00 | 86.96% | 1.37% | +1.0645 |
| healthcare_count | 1.00 | 0.00 | 52.17% | 0.82% | +1.0179 |
| education_count | 0.00 | 0.00 | 28.26% | 0.95% | +0.7341 |

SMD is a descriptive standardized mean difference between known station cells
and non-station cells. It is not a causal effect and should not be interpreted
as feature importance.

Only 46 positive cells are available, so
positive-group estimates are inherently noisy.

## Activity-Feature Redundancy

Pairs at or above an absolute Spearman correlation of
0.90:

- `poi_count_within_1000m` ↔ `transport_activity_within_1000m`: +0.9022

Nested local / 1-km / 2-km counts can be strongly correlated without being
mathematically identical. High correlation is a reason to prefer a parsimonious
feature set in downstream ML experiments.

## Population Overlap

Spearman correlation with the optional WorldPop context:
- `poi_count` strongest population association: `population_count` +0.3166
- `retail_commercial_count` strongest population association: `population_count` +0.2239
- `education_count` strongest population association: `population_count` +0.1858
- `healthcare_count` strongest population association: `population_count` +0.1751
- `transport_activity_count` strongest population association: `population_count` +0.2723
- `poi_count_within_1000m` strongest population association: `population_within_1000m` +0.4548
- `poi_count_within_2000m` strongest population association: `population_within_2000m` +0.5678
- `retail_commercial_within_1000m` strongest population association: `population_count` +0.3522
- `education_within_1000m` strongest population association: `population_count` +0.3417
- `healthcare_within_1000m` strongest population association: `population_count` +0.3072
- `transport_activity_within_1000m` strongest population association: `population_count` +0.4093

Population correlation is important because both feature families may encode
urbanization. A strong relationship does not make either feature invalid, but
it reduces the case for treating them as independent evidence.

## Interpretation Policy

OSM POI coverage is spatially heterogeneous. Zero or low counts may reflect
either low mapped urban activity or incomplete OSM mapping.

Activity counts are therefore treated as mapped urban-activity proxies rather
than direct observations of EV demand, trips, employment, retail turnover, or
traffic.

This audit is descriptive. The next evidence step is incremental evaluation
under the existing 5-km spatial block cross-validation design, with average
precision as the primary metric because the target is extremely imbalanced.

## Outputs

- `data/processed/ankara_activity_feature_audit_distributions.csv`
- `data/processed/ankara_activity_feature_target_comparison.csv`
- `data/processed/ankara_activity_feature_correlations.csv`
- `data/processed/ankara_activity_feature_redundancy_pairs.csv`
- `docs/ankara_activity_feature_correlations.png`

## Generated At

2026-08-11T12:37:37.454851+00:00
