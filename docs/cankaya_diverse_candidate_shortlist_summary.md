# Çankaya Diverse Candidate Shortlist

## Source

- Source candidate scores: `cankaya_candidate_suitability_scores.csv`
- Total scored candidates: 7,217
- Eligible candidates after score thresholds: 346
- Selected candidates: 20
- Generated at: 2026-08-01T12:35:48.191620+00:00

## Eligibility Rules

A grid cell must satisfy all of the following conditions before spatial
selection:

- Suitability score: at least 60/100
- Feasibility score: at least 60/100
- Need score: at least 50/100

Individual criterion failures in the complete score dataset:

- Below suitability threshold: 6,722
- Below feasibility threshold: 6,112
- Below need threshold: 1,556

These counts overlap because a candidate can fail more than one rule.

## Spatial Selection

Eligible candidates are ordered by their original suitability rank.
The highest-ranked candidate is selected first. Each following
candidate is accepted only when its grid centroid is at least
1,000 metres from every previously selected
candidate.

This greedy procedure continues until
20 candidates are selected.

## Result Statistics

- Minimum selected-candidate spacing: 1,000.00 metres
- Maximum nearest-selected spacing: 10,514.87 metres
- Worst original suitability rank selected: 114
- Lowest selected suitability score: 67.40
- Lowest selected feasibility score: 70.57

## Selected Candidates

|   diverse_selection_rank |   suitability_rank | grid_id   |   suitability_score |   feasibility_score |   need_score |   accessibility_score |   parking_score |   infrastructure_gap_score |   nearest_selected_candidate_m |   center_latitude |   center_longitude |
|-------------------------:|-------------------:|:----------|--------------------:|--------------------:|-------------:|----------------------:|----------------:|---------------------------:|-------------------------------:|------------------:|-------------------:|
|                        1 |                  1 | CKY_00162 |               77.39 |               96.37 |        62.15 |                 95.89 |           97.10 |                      55.47 |                        1000.00 |             39.87 |              32.65 |
|                        2 |                  2 | CKY_00428 |               76.54 |               85.63 |        68.42 |                 86.37 |           84.54 |                      62.85 |                        1118.03 |             39.85 |              32.67 |
|                        3 |                  4 | CKY_00190 |               76.30 |               92.27 |        63.09 |                 89.21 |           96.87 |                      56.58 |                        1060.66 |             39.85 |              32.66 |
|                        4 |                  9 | CKY_00501 |               74.69 |               87.20 |        63.98 |                 93.45 |           77.84 |                      57.62 |                        1118.03 |             39.86 |              32.68 |
|                        5 |                 11 | CKY_00054 |               74.38 |               96.00 |        57.63 |                 95.51 |           96.74 |                      50.15 |                        1060.66 |             39.86 |              32.64 |
|                        6 |                 18 | CKY_00322 |               72.62 |               87.89 |        60.01 |                 89.06 |           86.13 |                      52.95 |                        1030.78 |             39.87 |              32.67 |
|                        7 |                 22 | CKY_01951 |               72.46 |               78.46 |        66.93 |                 97.95 |           49.21 |                      61.09 |                        3288.24 |             39.81 |              32.80 |
|                        8 |                 24 | CKY_00158 |               72.15 |               83.16 |        62.60 |                 80.61 |           86.99 |                      56.00 |                        1000.00 |             39.87 |              32.65 |
|                        9 |                 33 | CKY_02773 |               71.25 |               81.29 |        62.46 |                 79.21 |           84.40 |                      55.83 |                        3010.40 |             39.83 |              32.85 |
|                       10 |                 70 | CKY_00743 |               68.72 |               84.12 |        56.13 |                 95.11 |           67.63 |                      48.39 |                        2304.89 |             39.87 |              32.70 |
|                       11 |                 75 | CKY_00284 |               68.48 |               85.90 |        54.59 |                 92.33 |           76.26 |                      46.57 |                        1250.00 |             39.88 |              32.66 |
|                       12 |                 83 | CKY_01791 |               68.33 |               74.93 |        62.31 |                 90.41 |           51.71 |                      55.66 |                        1500.00 |             39.84 |              32.79 |
|                       13 |                 84 | CKY_02059 |               68.33 |               71.39 |        65.40 |                 99.72 |           28.89 |                      59.29 |                        1500.00 |             39.84 |              32.81 |
|                       14 |                 85 | CKY_00181 |               68.29 |               70.57 |        66.09 |                 81.02 |           54.89 |                      60.10 |                        1677.05 |             39.83 |              32.66 |
|                       15 |                 87 | CKY_00027 |               68.26 |               82.88 |        56.22 |                 93.65 |           66.72 |                      48.50 |                        1118.03 |             39.87 |              32.64 |
|                       16 |                 94 | CKY_00081 |               68.00 |               77.57 |        59.61 |                 81.79 |           71.23 |                      52.49 |                        1060.66 |             39.85 |              32.65 |
|                       17 |                 97 | CKY_02382 |               67.93 |               82.70 |        55.79 |                 89.75 |           72.14 |                      47.99 |                        1820.03 |             39.85 |              32.83 |
|                       18 |                111 | CKY_01232 |               67.45 |               82.50 |        55.15 |                 97.03 |           60.71 |                      47.24 |                        2150.58 |             39.89 |              32.74 |
|                       19 |                112 | CKY_03429 |               67.45 |               75.30 |        60.42 |                 90.94 |           51.83 |                      53.43 |                       10514.87 |             39.93 |              32.88 |
|                       20 |                114 | CKY_01481 |               67.40 |               84.33 |        53.87 |                 95.93 |           66.94 |                      45.73 |                        2150.58 |             39.88 |              32.76 |

## Interpretation

The complete suitability dataset should be used for continuous map
visualization and detailed analysis. This shortlist is intended for
field review, stakeholder evaluation and preliminary feasibility
assessment.

The shortlist does not represent final installation decisions.
Candidate polygons still require on-site validation, electrical-grid
capacity checks, ownership and permit review, traffic analysis and
verified demand data.
