# Ankara Diverse Candidate Shortlist

## Selection Configuration

- Total scored candidates: 102,699
- Eligible candidates: 4,954
- Desired shortlist size: 20
- Minimum spatial separation: 25,000 m
- Minimum suitability score: 60
- Minimum feasibility score: 60
- Minimum need score: 50

## Final Shortlist

- Selected candidates: 20
- Minimum observed spacing: 25,079.87 m
- Best original suitability rank selected: 1
- Worst original suitability rank selected: 8,728
- Lowest selected suitability: 63.1078
- Lowest selected feasibility: 60.2973
- Lowest selected need: 54.3497

## Selected Candidates

- #1: `ANK_004300` — suitability 89.75, feasibility 82.80, need 97.28
- #2: `ANK_055975` — suitability 84.85, feasibility 88.56, need 81.30
- #3: `ANK_007151` — suitability 84.20, feasibility 74.74, need 94.86
- #4: `ANK_013005` — suitability 83.33, feasibility 75.72, need 91.71
- #5: `ANK_101065` — suitability 81.50, feasibility 81.92, need 81.08
- #6: `ANK_030609` — suitability 78.67, feasibility 81.74, need 75.71
- #7: `ANK_037238` — suitability 78.03, feasibility 68.03, need 89.50
- #8: `ANK_044917` — suitability 76.70, feasibility 84.47, need 69.64
- #9: `ANK_093363` — suitability 73.71, feasibility 77.61, need 69.99
- #10: `ANK_066425` — suitability 73.54, feasibility 60.73, need 89.05
- #11: `ANK_102631` — suitability 73.38, feasibility 75.00, need 71.79
- #12: `ANK_057005` — suitability 72.20, feasibility 91.40, need 57.03
- #13: `ANK_061180` — suitability 71.04, feasibility 92.86, need 54.35
- #14: `ANK_010010` — suitability 70.32, feasibility 61.36, need 80.60
- #15: `ANK_073387` — suitability 68.40, feasibility 71.91, need 65.06
- #16: `ANK_075344` — suitability 67.68, feasibility 61.59, need 74.38
- #17: `ANK_093670` — suitability 67.43, feasibility 60.30, need 75.41
- #18: `ANK_080777` — suitability 66.65, feasibility 66.92, need 66.37
- #19: `ANK_028141` — suitability 66.31, feasibility 66.42, need 66.20
- #20: `ANK_010930` — suitability 63.11, feasibility 67.52, need 58.98

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

2026-08-11T07:53:43.073231+00:00
