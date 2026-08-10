# Ankara Baseline Model Comparison

## Shared Evaluation Design

- Rows: 102,745
- Positive station cells: 46
- Spatial block size: 5 km
- Cross-validation folds: 5
- Predictor set: identical leakage-safe road and parking features

Both models are evaluated using exactly the same predefined spatial
fold assignments.

## Logistic Regression

- Pooled AP: 0.031053
- Mean fold AP: 0.050466
- Fold AP std: 0.024544
- Pooled ROC-AUC: 0.967786
- Top 1% recall: 0.543478
- Top 5% recall: 0.891304

## Random Forest

- Pooled AP: 0.050246
- Mean fold AP: 0.062796
- Fold AP std: 0.060769
- Pooled ROC-AUC: 0.960337
- Top 1% recall: 0.369565
- Top 5% recall: 0.934783

## Interpretation

Random Forest produces the stronger pooled average precision and
slightly stronger top-5-percent retrieval.

Logistic Regression produces substantially lower fold-to-fold AP
variation and stronger retrieval within the highest-ranked one percent
of grid cells.

Random Forest therefore does not unambiguously replace the Logistic
Regression baseline.

The two models capture partially different ranking behavior.

## OOF Ranking Agreement

- Spearman score correlation: 0.474717

## Model-Selection Policy

No final model is selected using accuracy.

Primary criteria are:

- pooled average precision
- fold-level average precision stability
- top-1-percent recall
- top-5-percent recall
- lift over class prevalence

ROC-AUC is retained as a secondary ranking metric.

## Important Limitations

Only 46 positive station cells are available.

Performance differences are therefore sensitive to the geographic
distribution of rare positive examples.

Random Forest impurity importance and Logistic Regression coefficients
are descriptive rather than causal.

## Outputs

- `data/processed/ankara_baseline_model_comparison.csv`
- `data/processed/ankara_baseline_combined_oof_predictions.csv`
- `docs/ankara_baseline_model_comparison_pr_curve.png`

## Generated At

2026-08-10T09:03:45.292020+00:00
