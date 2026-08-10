# Ankara Diverse Candidate Shortlist

## Selection Configuration

- Total scored candidates: 102,699
- Eligible candidates: 10,770
- Desired shortlist size: 20
- Minimum spatial separation: 25,000 m
- Minimum suitability score: 60
- Minimum feasibility score: 60
- Minimum need score: 50

## Final Shortlist

- Selected candidates: 20
- Minimum observed spacing: 25,104.78 m
- Best original suitability rank selected: 1
- Worst original suitability rank selected: 5,815
- Lowest selected suitability: 70.4635
- Lowest selected feasibility: 68.9182
- Lowest selected need: 56.0840

## Selected Candidates

- #1: `ANK_004429` — suitability 92.55, feasibility 98.87, need 86.65
- #2: `ANK_012588` — suitability 87.81, feasibility 94.01, need 82.01
- #3: `ANK_045529` — suitability 84.96, feasibility 91.95, need 78.50
- #4: `ANK_101065` — suitability 82.92, feasibility 97.40, need 70.59
- #5: `ANK_001267` — suitability 80.43, feasibility 72.88, need 88.76
- #6: `ANK_030609` — suitability 79.87, feasibility 97.81, need 65.23
- #7: `ANK_008258` — suitability 77.30, feasibility 74.98, need 79.70
- #8: `ANK_019391` — suitability 77.14, feasibility 71.80, need 82.89
- #9: `ANK_058928` — suitability 76.55, feasibility 73.11, need 80.16
- #10: `ANK_057142` — suitability 76.55, feasibility 71.58, need 81.86
- #11: `ANK_044917` — suitability 76.39, feasibility 98.64, need 59.16
- #12: `ANK_044069` — suitability 75.03, feasibility 68.92, need 81.68
- #13: `ANK_093363` — suitability 74.79, feasibility 94.01, need 59.51
- #14: `ANK_058117` — suitability 74.01, feasibility 97.68, need 56.08
- #15: `ANK_070064` — suitability 73.94, feasibility 70.90, need 77.10
- #16: `ANK_100200` — suitability 73.50, feasibility 70.30, need 76.84
- #17: `ANK_102675` — suitability 73.11, feasibility 84.88, need 62.97
- #18: `ANK_087378` — suitability 72.17, feasibility 69.93, need 74.47
- #19: `ANK_019754` — suitability 72.07, feasibility 68.97, need 75.31
- #20: `ANK_070026` — suitability 70.46, feasibility 80.71, need 61.52

## Method

Candidates first pass the same suitability, feasibility and need
quality filters used by the Çankaya pilot.

The remaining Ankara candidates are ordered by their original
suitability rank. A greedy spatial-diversity rule then selects the
highest-ranked candidate whose representative point is at least
25 kilometres from every already-selected candidate.

The Ankara spacing threshold is intentionally larger than the
1-kilometre Çankaya pilot threshold because the province-wide study
area is substantially larger.

The spatial-diversity rule changes only the final shortlist. It does
not change any candidate's underlying suitability score.

## Outputs

- `data/processed/ankara_diverse_candidate_shortlist.csv`
- `data/processed/ankara_diverse_candidate_shortlist.gpkg`

## Generated At

2026-08-10T07:39:17.810973+00:00
