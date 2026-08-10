# Ankara Model Ranking Comparison

## Evaluation Design

- Rows: 102,745
- Positive station cells: 46
- Spatial folds: 5
- Spatial block size: 5 km

## Fold-Normalized Ranking

Each base model was independently trained inside each spatial
cross-validation iteration.

Absolute score scales can therefore differ across folds.

Before pooled top-ranked retrieval analysis, every model score is
converted to a percentile rank inside its own validation fold.

This normalization preserves within-fold ordering while reducing the
effect of incompatible absolute score scales across independently
fitted fold models.

## Models

- Logistic Regression: pooled AP 0.031570, mean fold AP 0.050466, AP std 0.024544, top-1% recall 0.5435, top-5% recall 0.8696
- Random Forest: pooled AP 0.032100, mean fold AP 0.062796, AP std 0.060769, top-1% recall 0.3696, top-5% recall 0.9130
- Gradient Boosting: pooled AP 0.047518, mean fold AP 0.091578, AP std 0.078839, top-1% recall 0.4130, top-5% recall 0.8261
- Unweighted Rank Ensemble: pooled AP 0.057249, mean fold AP 0.094922, AP std 0.084620, top-1% recall 0.5000, top-5% recall 0.9130

## Ensemble

The ensemble is a fixed equal-weight mean of the three fold-normalized
model ranks.

No model-specific ensemble weights were tuned.

This avoids selecting weights against only 46 positive observations.

The ensemble should be interpreted as an exploratory ranking
combination rather than an independently validated production model.

## Base-Model Rank Correlation

                     Logistic Regression  Random Forest  Gradient Boosting
Logistic Regression             1.000000       0.276175           0.334942
Random Forest                   0.276175       1.000000           0.386513
Gradient Boosting               0.334942       0.386513           1.000000

Lower rank agreement indicates that models capture partially different
patterns and may therefore contain complementary ranking information.

## Important Limitation

The same spatial OOF predictions are used to describe the individual
models and the fixed ensemble.

Although ensemble weights are not tuned, the ensemble analysis is still
exploratory and should not be treated as an independent external
validation result.

## Outputs

- `data/processed/ankara_model_ranking_comparison.csv`
- `data/processed/ankara_model_rank_ensemble_oof.csv`
- `data/processed/ankara_model_rank_correlation.csv`
- `docs/ankara_model_rank_comparison_pr_curve.png`

## Generated At

2026-08-10T09:17:39.084031+00:00
