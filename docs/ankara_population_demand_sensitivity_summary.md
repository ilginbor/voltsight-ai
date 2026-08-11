# Ankara Population Demand Sensitivity

## Purpose

This diagnostic studies how a separate residential-population demand pillar
behaves before it is allowed to change the canonical suitability formula.

The existing feasibility, need, suitability scores, ranks, and shortlist are
not modified by this script.

## Candidate Data

- Candidate rows: 102,699
- Population-positive local cells: 42,579
- Local population median: 0.00
- 1-km population median: 4.39
- 2-km population median: 78.39

## Zero-Preserving Demand Components

Three population variables are converted to positive-only percentile scores:

- `local_population_score`
- `population_1km_score`
- `population_2km_score`

True zero values remain zero. Positive population values are ranked only
against other positive values.

`population_density_per_km2` is not scored separately because it is a
deterministic scale transform of `population_count` on the fixed 500-m grid.

## Weight Scenarios

- `local_only`: 100% local
- `near_context`: 30% local, 40% within 1 km, 30% within 2 km
- `balanced_context`: 20% local, 35% within 1 km, 45% within 2 km
- `broad_context`: 10% local, 30% within 1 km, 60% within 2 km

The scenarios are sensitivity diagnostics, not fitted or optimized weights.

## Scenario Results

| Scenario | Weights | Median demand | Spearman vs feasibility | Spearman vs need | Spearman vs suitability | Top 1% overlap | Top 5% overlap | Current top-20 median demand | Top-20 demand >=70 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local_only | Local only | 0.0000 | 0.5445 | -0.3361 | 0.4099 | 0.00% | 3.88% | 89.3175 | 15/20 |
| near_context | 30% local / 40% 1 km / 30% 2 km | 33.4433 | 0.6681 | -0.4396 | 0.4916 | 0.00% | 2.57% | 91.7804 | 17/20 |
| balanced_context | 20% local / 35% 1 km / 45% 2 km | 37.9038 | 0.6661 | -0.4481 | 0.4855 | 0.00% | 2.28% | 91.9905 | 17/20 |
| broad_context | 10% local / 30% 1 km / 60% 2 km | 42.2971 | 0.6581 | -0.4539 | 0.4744 | 0.00% | 1.60% | 92.1336 | 17/20 |

## Correlation Diagnostics

- Minimum pairwise Spearman correlation among the four demand scenarios:
  0.6706
- Minimum pairwise Spearman correlation among local/1-km/2-km demand
  components: 0.5755
- Maximum pairwise Spearman correlation among local/1-km/2-km demand
  components: 0.8507

For the reference `balanced_context` scenario:

- Spearman with feasibility: 0.6661
- Spearman with need: -0.4481
- Spearman with current suitability: 0.4855
- Current suitability top-1% overlap: 0.00%
- Current suitability top-5% overlap: 2.28%
- Median demand score among the current suitability top 20:
  91.9905
- Current suitability top-20 cells with demand >= 70:
  17/20

## Interpretation Policy

High correlation between demand scenarios means exact local-versus-neighborhood
weights have limited effect on province-wide demand ordering. Lower correlation
means the weight choice materially changes which cells are described as
high-demand.

Correlation with feasibility or current suitability indicates overlap with the
existing road/parking decision layer. Correlation with need indicates overlap
with the charging-gap layer. These are descriptive associations, not causal
effects.

Top-fraction overlap describes whether current suitability and population
demand prioritize the same cells. Low overlap does not automatically mean one
ranking is wrong; it can indicate that demand contributes a distinct decision
dimension.

This diagnostic intentionally does not create a new final suitability score.
A top-level demand weight should only be selected after reviewing these
redundancy and ranking-stability results.

Population remains a modeled residential-demand proxy and does not directly
measure traffic, employment, commuting, retail activity, tourism, EV
ownership, or distribution-grid capacity.

## Outputs

- `data/processed/ankara_population_demand_sensitivity.csv`
- `data/processed/ankara_population_demand_correlations.csv`
- `data/processed/ankara_population_demand_scenario_metrics.csv`
- `docs/ankara_population_demand_sensitivity.png`

## Generated At

2026-08-11T11:45:53.833331+00:00
