# Ankara Candidate Suitability Summary

## Candidates

- Candidate grid cells: 102,699
- Median suitability score: 45.63
- Maximum suitability score: 92.55
- Minimum suitability score: 1.88
- Top candidate: `ANK_004429`
- Top candidate score: 92.5548

## Priority Bands

- A: 1,027
- B: 4,108
- C: 15,405
- D: 30,810
- E: 51,349

## Scoring Model

### Accessibility

- Main-road proximity: 45%
- Main-road presence: 35%
- Road density: 20%

### Parking

- Nearest-parking proximity: 45%
- Parking within 1 km: 35%
- Local parking area: 20%

### Infrastructure Gap

- Distance to nearest charging station: 75%
- Charging-station scarcity within 2 km: 25%

### Technology Gap

- DC absence within 1 km: 60%
- AC absence within 1 km: 40%

### Composite Scores

- Feasibility = 60% accessibility + 40% parking
- Need = 85% infrastructure gap + 15% technology gap
- Suitability = geometric mean of feasibility and need

## Interpretation

The score is an explainable decision-support ranking rather than a
probability that a charging station should be constructed.

Percentile transformations are calculated over Ankara candidate cells,
so scores are relative to the province-wide candidate distribution.

## Outputs

- `data/processed/ankara_candidate_suitability_scores.csv`
- `data/processed/ankara_candidate_suitability_scores.gpkg`

## Generated At

2026-08-10T06:30:32.583418+00:00
