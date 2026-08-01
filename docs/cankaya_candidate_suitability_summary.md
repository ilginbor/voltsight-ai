# Çankaya Candidate Suitability Score Summary

## Source

- Candidate source: `cankaya_candidate_site_dataset.csv`
- Candidate rows: 7,217
- Generated at: 2026-08-01T12:30:51.161155+00:00
- Output CSV: `data/processed/cankaya_candidate_suitability_scores.csv`
- Output GeoPackage: `data/processed/cankaya_candidate_suitability_scores.gpkg`

## Method

This is an explainable decision-support score, not a trained machine
learning prediction.

Feature percentiles are calculated relative to the current Çankaya
candidate grid population. Zero-inflated road and parking quantities
retain a score of zero when the underlying quantity is zero.

### Accessibility Score

- 45% proximity to a main road
- 35% main-road length inside the grid cell
- 20% road density

### Parking Score

- 45% proximity to the nearest mapped parking feature
- 35% parking count within 1,000 metres
- 20% mapped parking area inside the grid cell

### Infrastructure Gap Score

- 75% distance from the nearest existing charging station
- 25% scarcity of charging stations within 2,000 metres

### Technology Gap Score

- 60% absence of a mapped DC station within 1,000 metres
- 40% absence of a mapped AC station within 1,000 metres

### Combined Scores

- Feasibility = 60% accessibility + 40% parking
- Need = 85% infrastructure gap + 15% technology gap
- Suitability = square root of feasibility multiplied by need

The geometric combination prevents remote cells with a large
infrastructure gap but very poor road and parking feasibility from
automatically receiving the highest rankings.

## Priority Bands

- A - Highest priority: 73
- B - High priority: 288
- C - Medium priority: 1,083
- D - Lower priority: 2,165
- E - Lowest priority: 3,608

Bands are relative rankings:

- A: top 1%
- B: 95th to below 99th percentile
- C: 80th to below 95th percentile
- D: 50th to below 80th percentile
- E: below the 50th percentile

## Score Distribution

|                          |   min |    25% |    50% |    75% |    max |
|:-------------------------|------:|-------:|-------:|-------:|-------:|
| accessibility_score      |  0.01 |  11.7  |  24.24 |  47.58 |  99.82 |
| parking_score            |  0.01 |  11.25 |  22.5  |  48.93 |  99.31 |
| infrastructure_gap_score |  5.67 |  43.76 |  62.51 |  81.25 | 100    |
| technology_gap_score     |  0    | 100    | 100    | 100    | 100    |
| feasibility_score        |  0.74 |  12.52 |  24.38 |  47.96 |  96.37 |
| need_score               |  4.82 |  52.19 |  68.13 |  84.06 | 100    |
| suitability_score        |  8.6  |  30.46 |  38.3  |  46.99 |  77.39 |

## Top 20 Candidate Grid Cells

|   suitability_rank | grid_id   | district   |   suitability_score |   feasibility_score |   need_score |   accessibility_score |   parking_score |   infrastructure_gap_score |   technology_gap_score | priority_band        |
|-------------------:|:----------|:-----------|--------------------:|--------------------:|-------------:|----------------------:|----------------:|---------------------------:|-----------------------:|:---------------------|
|                  1 | CKY_00162 | Çankaya    |               77.39 |               96.37 |        62.15 |                 95.89 |           97.10 |                      55.47 |                 100.00 | A - Highest priority |
|                  2 | CKY_00428 | Çankaya    |               76.54 |               85.63 |        68.42 |                 86.37 |           84.54 |                      62.85 |                 100.00 | A - Highest priority |
|                  3 | CKY_00312 | Çankaya    |               76.48 |               87.45 |        66.88 |                 89.02 |           85.09 |                      61.04 |                 100.00 | A - Highest priority |
|                  4 | CKY_00190 | Çankaya    |               76.30 |               92.27 |        63.09 |                 89.21 |           96.87 |                      56.58 |                 100.00 | A - Highest priority |
|                  5 | CKY_00059 | Çankaya    |               75.59 |               95.42 |        59.89 |                 97.06 |           92.96 |                      52.81 |                 100.00 | A - Highest priority |
|                  6 | CKY_00272 | Çankaya    |               75.58 |               87.18 |        65.53 |                 88.44 |           85.28 |                      59.45 |                 100.00 | A - Highest priority |
|                  7 | CKY_00499 | Çankaya    |               74.89 |               84.64 |        66.27 |                 91.09 |           74.97 |                      60.31 |                 100.00 | A - Highest priority |
|                  8 | CKY_00116 | Çankaya    |               74.72 |               91.92 |        60.74 |                 93.83 |           89.05 |                      53.81 |                 100.00 | A - Highest priority |
|                  9 | CKY_00501 | Çankaya    |               74.69 |               87.20 |        63.98 |                 93.45 |           77.84 |                      57.62 |                 100.00 | A - Highest priority |
|                 10 | CKY_00229 | Çankaya    |               74.57 |               86.46 |        64.31 |                 87.26 |           85.26 |                      58.02 |                 100.00 | A - Highest priority |
|                 11 | CKY_00054 | Çankaya    |               74.38 |               96.00 |        57.63 |                 95.51 |           96.74 |                      50.15 |                 100.00 | A - Highest priority |
|                 12 | CKY_00155 | Çankaya    |               74.35 |               89.26 |        61.93 |                 89.46 |           88.97 |                      55.21 |                 100.00 | A - Highest priority |
|                 13 | CKY_00119 | Çankaya    |               73.78 |               90.08 |        60.44 |                 89.61 |           90.79 |                      53.45 |                 100.00 | A - Highest priority |
|                 14 | CKY_00152 | Çankaya    |               73.70 |               87.72 |        61.92 |                 87.40 |           88.20 |                      55.20 |                 100.00 | A - Highest priority |
|                 15 | CKY_00270 | Çankaya    |               73.63 |               82.65 |        65.59 |                 88.14 |           74.40 |                      59.52 |                 100.00 | A - Highest priority |
|                 16 | CKY_00468 | Çankaya    |               73.25 |               83.60 |        64.19 |                 80.16 |           88.75 |                      57.87 |                 100.00 | A - Highest priority |
|                 17 | CKY_00395 | Çankaya    |               73.04 |               80.93 |        65.91 |                 84.01 |           76.32 |                      59.90 |                 100.00 | A - Highest priority |
|                 18 | CKY_00322 | Çankaya    |               72.62 |               87.89 |        60.01 |                 89.06 |           86.13 |                      52.95 |                 100.00 | A - Highest priority |
|                 19 | CKY_00151 | Çankaya    |               72.57 |               84.93 |        62.01 |                 88.23 |           79.99 |                      55.30 |                 100.00 | A - Highest priority |
|                 20 | CKY_00500 | Çankaya    |               72.53 |               80.73 |        65.16 |                 79.34 |           82.82 |                      59.01 |                 100.00 | A - Highest priority |

## Important Limitations

The score reflects mapped OpenStreetMap, EPDK, road and parking
coverage. Missing map objects do not necessarily mean that real-world
infrastructure is absent.

The weights are explicit expert assumptions derived from the observed
feature distributions and correlations. They should later be tested
with stakeholder feedback, utilization data, population, traffic,
electric-grid capacity and verified installation outcomes.
