# Ankara Population-Suitability Blend Sensitivity

## Purpose

This diagnostic tests how much the current Ankara suitability ranking would
move if a small residential-demand contribution were added at the top level.

The canonical suitability formula and canonical shortlist are not modified by
this script.

## Inputs

- Candidate rows: 102,699
- Current suitability score: existing geometric feasibility/need score
- Demand score: `balanced_context`
- Demand score median: 37.9038
- Demand score maximum: 99.9742

## Diagnostic Blend

For demand weight `w`:

`diagnostic_score = (1 - w) * current_suitability + w * demand`

Weights tested:

- 0% demand / 100% current suitability
- 5% demand / 95% current suitability
- 10% demand / 90% current suitability
- 15% demand / 85% current suitability
- 20% demand / 80% current suitability

This convex blend is deliberately simple and interpretable. It is a
sensitivity device, not a claim that the final production score must be an
arithmetic blend.

## Results

| Demand weight | Suitability weight | Spearman vs baseline | Top 1% overlap | Top 5% overlap | Top-20 overlap | Median abs rank shift | P95 abs rank shift | Top-20 median suitability | Top-20 median feasibility | Top-20 median need | Top-20 median demand | Demand >=70 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 100% | 1.0000 | 100.00% | 100.00% | 100.00% | 0 | 0 | 86.3265 | 77.0913 | 97.1455 | 91.9905 | 17/20 |
| 5% | 95% | 0.9941 | 84.91% | 92.85% | 85.00% | 1,648 | 6,888 | 86.3265 | 76.9373 | 97.1455 | 92.6540 | 20/20 |
| 10% | 90% | 0.9783 | 71.86% | 85.14% | 85.00% | 3,319 | 13,101 | 86.3265 | 76.9373 | 97.1455 | 92.6540 | 20/20 |
| 15% | 85% | 0.9557 | 59.88% | 76.48% | 80.00% | 5,006 | 18,491 | 86.3265 | 76.9373 | 97.1455 | 92.7687 | 20/20 |
| 20% | 80% | 0.9291 | 51.61% | 65.28% | 75.00% | 6,655 | 22,986 | 86.3265 | 76.9373 | 97.1309 | 92.8287 | 20/20 |

## Interpretation Policy

The zero-demand row must reproduce the current suitability ranking exactly.

Spearman correlation measures province-wide ranking stability. Top-1%, top-5%,
and top-20 overlap focus on decision-relevant high-ranked cells.

Rank-shift diagnostics show whether a small demand contribution only reorders
the top of the list or materially reshuffles the wider candidate universe.

The top-20 feasibility, need, and demand summaries expose the central trade-off:
population demand is positively associated with road/parking feasibility but
negatively associated with the existing charging-gap need score.

A demand weight should not be selected merely because it increases residential
demand among top candidates. The selected weight should preserve meaningful
infrastructure-need coverage and avoid turning a province-wide infrastructure
planning score into an urban-population ranking.

Population remains a modeled residential-demand proxy. It does not directly
measure traffic, employment, commuting, retail activity, tourism, EV ownership,
or distribution-grid capacity.

This analysis is descriptive decision-model sensitivity, not ML validation and
not causal inference.

## Outputs

- `data/processed/ankara_population_suitability_blend_sensitivity.csv`
- `data/processed/ankara_population_suitability_blend_metrics.csv`
- `docs/ankara_population_suitability_blend_sensitivity.png`

## Generated At

2026-08-11T11:51:59.504877+00:00
