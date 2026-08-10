# Ankara Diverse Candidate Shortlist

## Selection Configuration

- Total scored candidates: 102,699
- Eligible candidates: 10,770
- Desired shortlist size: 20
- Minimum spatial separation: 5,000 m
- Minimum suitability score: 60
- Minimum feasibility score: 60
- Minimum need score: 50

## Final Shortlist

- Selected candidates: 20
- Minimum observed spacing: 5,000.00 m
- Best original suitability rank selected: 1
- Worst original suitability rank selected: 205
- Lowest selected suitability: 82.9348
- Lowest selected feasibility: 78.2816
- Lowest selected need: 70.8192

## Selected Candidates

- #1: `ANK_004429` — suitability 92.55, feasibility 98.87, need 86.65
- #2: `ANK_006983` — suitability 89.00, feasibility 93.81, need 84.44
- #3: `ANK_012588` — suitability 87.81, feasibility 94.01, need 82.01
- #4: `ANK_003788` — suitability 85.74, feasibility 84.51, need 86.99
- #5: `ANK_003799` — suitability 85.04, feasibility 82.79, need 87.35
- #6: `ANK_045529` — suitability 84.96, feasibility 91.95, need 78.50
- #7: `ANK_004445` — suitability 84.88, feasibility 82.60, need 87.22
- #8: `ANK_004286` — suitability 84.48, feasibility 82.70, need 86.31
- #9: `ANK_005033` — suitability 84.48, feasibility 82.83, need 86.16
- #10: `ANK_006359` — suitability 84.22, feasibility 83.55, need 84.90
- #11: `ANK_049259` — suitability 84.20, feasibility 92.24, need 76.85
- #12: `ANK_007663` — suitability 83.99, feasibility 84.07, need 83.90
- #13: `ANK_010892` — suitability 83.57, feasibility 83.80, need 83.34
- #14: `ANK_012017` — suitability 83.49, feasibility 83.92, need 83.05
- #15: `ANK_055975` — suitability 83.43, feasibility 98.29, need 70.82
- #16: `ANK_003592` — suitability 83.36, feasibility 80.13, need 86.72
- #17: `ANK_002980` — suitability 83.29, feasibility 79.47, need 87.29
- #18: `ANK_007613` — suitability 83.08, feasibility 81.54, need 84.66
- #19: `ANK_002997` — suitability 82.97, feasibility 78.28, need 87.94
- #20: `ANK_006429` — suitability 82.93, feasibility 80.53, need 85.42

## Method

Candidates first pass the same suitability, feasibility and need
quality filters used by the Çankaya pilot.

The remaining Ankara candidates are ordered by their original
suitability rank. A greedy spatial-diversity rule then selects the
highest-ranked candidate whose representative point is at least
5 kilometres from every already-selected candidate.

The Ankara spacing threshold is intentionally larger than the
1-kilometre Çankaya pilot threshold because the province-wide study
area is substantially larger.

The spatial-diversity rule changes only the final shortlist. It does
not change any candidate's underlying suitability score.

## Outputs

- `data/processed/ankara_diverse_candidate_shortlist.csv`
- `data/processed/ankara_diverse_candidate_shortlist.gpkg`

## Generated At

2026-08-10T06:34:23.411694+00:00
