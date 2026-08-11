# Ankara Decision-Support Export

## Purpose

This export packages the canonical 20-site Ankara shortlist into a stable,
frontend/API-oriented contract.

The export does not create a new score and does not blend machine-learning
predictions into suitability.

## Inputs

- `data/processed/ankara_diverse_candidate_shortlist.csv`
- `data/processed/ankara_shortlist_ml_support.csv`

## Outputs

- `data/processed/ankara_decision_support_shortlist.csv`
- `data/processed/ankara_decision_support_shortlist.json`

The CSV is a flat analysis-friendly table.

The JSON is a nested application-oriented representation with:

- location
- explainable suitability components
- spatial-diversity diagnostics
- fold-normalized spatial OOF ML support

## Decision Policy

Primary decision layer:

- explainable suitability
- eligibility thresholds
- 25-km spatial diversity

Supporting evidence layer:

- Logistic Regression fold-normalized OOF percentile
- Random Forest fold-normalized OOF percentile
- HistGradientBoosting fold-normalized OOF percentile
- median cross-model consensus
- explicit cross-model disagreement information

ML support is not a calibrated probability and is not blended into the
canonical suitability score.

## Current Shortlist Export

- Rows: 20
- Median ML consensus percentile: 95.03
- All three models in candidate top 20%: 15/20
- Exactly two models in candidate top 20%: 5/20

## JSON Contract

Top-level keys:

```text
schema_version
study_area
study_area_country
coordinate_reference_system
candidate_count
decision_policy
generated_at_utc
candidates
```

Each candidate contains:

```text
grid_id
selection_rank
location
suitability
spatial_diversity
ml_support
```

## Interpretation

The application layer should present suitability and ML support as separate
axes.

A candidate with high suitability and lower ML agreement is not automatically
invalid. It can represent a gap-oriented recommendation that differs from the
historical mapped station-placement pattern.

Cross-model disagreement should remain visible rather than being hidden behind
the median consensus value.

## Generated At

2026-08-11T15:07:48.412227+00:00
