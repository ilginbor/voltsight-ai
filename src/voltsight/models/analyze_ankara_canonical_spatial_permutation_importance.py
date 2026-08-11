from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from voltsight.core.ankara_ml_features import (
    ACTIVITY_CONTEXT_FEATURE_COLUMNS,
    CANONICAL_ML_FEATURE_COLUMNS,
    NORMALIZED_12_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from voltsight.models.evaluate_ankara_canonical_ml_models import (
    MODEL_LABELS,
    build_model,
    fit_model,
    load_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_spatial_permutation_importance.csv"
)

FOLD_DROP_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_spatial_permutation_importance_fold_drops.csv"
)

PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_canonical_spatial_permutation_importance.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_canonical_spatial_permutation_importance_summary.md"
)

MODEL_NAMES = (
    "random_forest",
    "hist_gradient_boosting",
)

N_REPEATS = 5

MODEL_SEED_OFFSETS = {
    "random_forest": 0,
    "hist_gradient_boosting": 1_000_000,
}


def feature_group_for(
    feature: str,
) -> str:
    """Return the canonical feature-family label."""

    if feature in ACTIVITY_CONTEXT_FEATURE_COLUMNS:
        return "activity"

    if feature in NORMALIZED_12_FEATURE_COLUMNS:
        if feature.startswith(
            "parking_"
        ) or feature in {
            "distance_to_nearest_parking_m",
            "known_parking_capacity",
        }:
            return "parking"

        return "road"

    raise ValueError(
        f"Unknown canonical feature: {feature}"
    )


def validate_configuration() -> None:
    """Validate the fixed canonical permutation-importance design."""

    if len(
        CANONICAL_ML_FEATURE_COLUMNS
    ) != 15:
        raise ValueError(
            "Canonical feature set must contain 15 predictors."
        )

    if N_REPEATS < 1:
        raise ValueError(
            "At least one permutation repeat is required."
        )

    if set(
        MODEL_NAMES
    ) != {
        "random_forest",
        "hist_gradient_boosting",
    }:
        raise ValueError(
            "Canonical permutation importance expects RF and HGB."
        )

    grouped = {
        feature_group_for(
            feature
        )
        for feature in CANONICAL_ML_FEATURE_COLUMNS
    }

    if grouped != {
        "road",
        "parking",
        "activity",
    }:
        raise ValueError(
            "Canonical feature groups are incomplete."
        )


def permutation_seed(
    *,
    model_name: str,
    feature_index: int,
    fold: int,
    repeat: int,
) -> int:
    """Create deterministic, non-overlapping permutation seeds."""

    if model_name not in MODEL_SEED_OFFSETS:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    return int(
        MODEL_SEED_OFFSETS[
            model_name
        ]
        + feature_index * 100_000
        + fold * 1_000
        + repeat
        + 42
    )


def permute_feature_values(
    dataframe: pd.DataFrame,
    *,
    feature: str,
    random_state: int,
) -> pd.DataFrame:
    """Return a copy with one feature permuted inside the validation fold."""

    if feature not in dataframe.columns:
        raise ValueError(
            f"Feature not found: {feature}"
        )

    result = dataframe.copy()

    rng = np.random.default_rng(
        random_state
    )

    values = result[
        feature
    ].to_numpy(
        copy=True
    )

    result[
        feature
    ] = rng.permutation(
        values
    )

    return result


def calculate_ap(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Calculate average precision after validating finite inputs."""

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(
        np.unique(
            y_true
        )
    ) != 2:
        raise ValueError(
            "Both target classes are required for average precision."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Scores contain non-finite values."
        )

    return float(
        average_precision_score(
            y_true,
            scores,
        )
    )


def evaluate_model_permutation_importance(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Calculate validation-only spatial-fold permutation importance.

    Each model is fitted once per fold. For every canonical feature and repeat,
    only the validation values of that feature are shuffled. The model is not
    retrained after permutation.
    """

    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unsupported model for permutation importance: {model_name}"
        )

    feature_columns = list(
        CANONICAL_ML_FEATURE_COLUMNS
    )

    baseline_oof = np.full(
        len(
            dataframe
        ),
        np.nan,
        dtype=float,
    )

    baseline_fold_ap: dict[
        int,
        float,
    ] = {}

    permuted_oof: dict[
        tuple[
            str,
            int,
        ],
        np.ndarray,
    ] = {
        (
            feature,
            repeat,
        ): np.full(
            len(
                dataframe
            ),
            np.nan,
            dtype=float,
        )
        for feature in CANONICAL_ML_FEATURE_COLUMNS
        for repeat in range(
            N_REPEATS
        )
    }

    fold_records: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    folds = sorted(
        dataframe[
            "cv_fold"
        ].astype(
            int
        ).unique()
    )

    for fold in folds:
        train_mask = (
            dataframe[
                "cv_fold"
            ].to_numpy(
                dtype=int
            )
            != fold
        )

        validation_mask = (
            ~train_mask
        )

        train = dataframe.loc[
            train_mask
        ]

        validation = dataframe.loc[
            validation_mask
        ]

        y_train = train[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

        y_validation = validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

        model = build_model(
            model_name
        )

        fit_model(
            model_name,
            model,
            train[
                feature_columns
            ],
            y_train,
        )

        baseline_scores = (
            model.predict_proba(
                validation[
                    feature_columns
                ]
            )[
                :,
                1,
            ]
        )

        validation_positions = (
            np.flatnonzero(
                validation_mask
            )
        )

        baseline_oof[
            validation_positions
        ] = baseline_scores

        baseline_ap = calculate_ap(
            y_validation,
            baseline_scores,
        )

        baseline_fold_ap[
            fold
        ] = baseline_ap

        for feature_index, feature in enumerate(
            CANONICAL_ML_FEATURE_COLUMNS
        ):
            for repeat in range(
                N_REPEATS
            ):
                seed = permutation_seed(
                    model_name=model_name,
                    feature_index=feature_index,
                    fold=fold,
                    repeat=repeat,
                )

                permuted_validation = (
                    permute_feature_values(
                        validation[
                            feature_columns
                        ],
                        feature=feature,
                        random_state=seed,
                    )
                )

                permuted_scores = (
                    model.predict_proba(
                        permuted_validation
                    )[
                        :,
                        1,
                    ]
                )

                permuted_oof[
                    (
                        feature,
                        repeat,
                    )
                ][
                    validation_positions
                ] = permuted_scores

                permuted_ap = calculate_ap(
                    y_validation,
                    permuted_scores,
                )

                fold_records.append(
                    {
                        "model": model_name,
                        "model_label": MODEL_LABELS[
                            model_name
                        ],
                        "feature": feature,
                        "feature_group": (
                            feature_group_for(
                                feature
                            )
                        ),
                        "cv_fold": fold,
                        "repeat": repeat,
                        "baseline_average_precision": (
                            baseline_ap
                        ),
                        "permuted_average_precision": (
                            permuted_ap
                        ),
                        "average_precision_drop": (
                            baseline_ap
                            - permuted_ap
                        ),
                    }
                )

    if not np.isfinite(
        baseline_oof
    ).all():
        raise ValueError(
            "Baseline OOF predictions are incomplete."
        )

    y_true = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    baseline_pooled_ap = calculate_ap(
        y_true,
        baseline_oof,
    )

    records: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    fold_frame = pd.DataFrame(
        fold_records
    )

    for feature in CANONICAL_ML_FEATURE_COLUMNS:
        pooled_permuted_ap = []

        for repeat in range(
            N_REPEATS
        ):
            scores = permuted_oof[
                (
                    feature,
                    repeat,
                )
            ]

            if not np.isfinite(
                scores
            ).all():
                raise ValueError(
                    "Permuted OOF predictions are incomplete for "
                    f"{model_name} / {feature} / repeat {repeat}."
                )

            pooled_permuted_ap.append(
                calculate_ap(
                    y_true,
                    scores,
                )
            )

        pooled_permuted_ap_array = (
            np.asarray(
                pooled_permuted_ap,
                dtype=float,
            )
        )

        feature_fold_rows = (
            fold_frame.loc[
                fold_frame[
                    "feature"
                ]
                == feature
            ]
        )

        records.append(
            {
                "model": model_name,
                "model_label": (
                    MODEL_LABELS[
                        model_name
                    ]
                ),
                "feature": feature,
                "feature_group": (
                    feature_group_for(
                        feature
                    )
                ),
                "baseline_pooled_average_precision": (
                    baseline_pooled_ap
                ),
                "mean_permuted_pooled_average_precision": float(
                    pooled_permuted_ap_array.mean()
                ),
                "std_permuted_pooled_average_precision": float(
                    pooled_permuted_ap_array.std(
                        ddof=1
                    )
                ),
                "pooled_average_precision_drop": float(
                    baseline_pooled_ap
                    - pooled_permuted_ap_array.mean()
                ),
                "mean_fold_average_precision_drop": float(
                    feature_fold_rows[
                        "average_precision_drop"
                    ].mean()
                ),
                "std_fold_average_precision_drop": float(
                    feature_fold_rows[
                        "average_precision_drop"
                    ].std(
                        ddof=1
                    )
                ),
                "positive_fold_repeat_drop_fraction": float(
                    (
                        feature_fold_rows[
                            "average_precision_drop"
                        ]
                        > 0
                    ).mean()
                ),
                "repeat_count": N_REPEATS,
            }
        )

    importance = pd.DataFrame(
        records
    )

    importance = importance.sort_values(
        [
            "model",
            "pooled_average_precision_drop",
            "feature",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return (
        importance,
        fold_frame,
    )


def run_analysis(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run canonical permutation importance for RF and HGB."""

    validate_configuration()

    importance_frames = []
    fold_frames = []

    for model_name in MODEL_NAMES:
        print(
            "Running permutation importance:",
            MODEL_LABELS[
                model_name
            ],
        )

        (
            importance,
            fold_drops,
        ) = (
            evaluate_model_permutation_importance(
                dataframe,
                model_name=model_name,
            )
        )

        importance_frames.append(
            importance
        )

        fold_frames.append(
            fold_drops
        )

    return (
        pd.concat(
            importance_frames,
            ignore_index=True,
        ),
        pd.concat(
            fold_frames,
            ignore_index=True,
        ),
    )


def save_outputs(
    importance: pd.DataFrame,
    fold_drops: pd.DataFrame,
) -> None:
    """Save canonical permutation-importance outputs."""

    IMPORTANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
        encoding="utf-8",
    )

    fold_drops.to_csv(
        FOLD_DROP_PATH,
        index=False,
        encoding="utf-8",
    )


def create_plot(
    importance: pd.DataFrame,
) -> None:
    """Plot pooled AP drops for all canonical features and both models."""

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = importance.pivot(
        index="feature",
        columns="model_label",
        values="pooled_average_precision_drop",
    )

    order = (
        importance.groupby(
            "feature"
        )[
            "pooled_average_precision_drop"
        ]
        .mean()
        .sort_values(
            ascending=True
        )
        .index
    )

    pivot = pivot.loc[
        order
    ]

    axis = pivot.plot(
        kind="barh",
        figsize=(
            11,
            9,
        ),
    )

    axis.axvline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Pooled spatial OOF AP drop after validation-only permutation"
    )

    axis.set_ylabel(
        "Canonical Activity15 feature"
    )

    axis.set_title(
        "Ankara Canonical Activity15 Spatial Permutation Importance"
    )

    figure = axis.get_figure()

    figure.tight_layout()

    figure.savefig(
        PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_summary(
    importance: pd.DataFrame,
) -> None:
    """Write the canonical permutation-importance interpretation summary."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sections = []

    for model_name in MODEL_NAMES:
        model_rows = (
            importance.loc[
                importance[
                    "model"
                ]
                == model_name
            ]
            .sort_values(
                "pooled_average_precision_drop",
                ascending=False,
                kind="stable",
            )
        )

        lines = [
            "| Feature | Group | Pooled AP drop | Mean fold AP drop | Fold-drop std | Positive fold/repeat fraction |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]

        for row in model_rows.itertuples(
            index=False
        ):
            lines.append(
                "| "
                f"`{row.feature}` | "
                f"{row.feature_group} | "
                f"{row.pooled_average_precision_drop:+.6f} | "
                f"{row.mean_fold_average_precision_drop:+.6f} | "
                f"{row.std_fold_average_precision_drop:.6f} | "
                f"{row.positive_fold_repeat_drop_fraction:.2%} |"
            )

        sections.append(
            "## "
            + MODEL_LABELS[
                model_name
            ]
            + "\n\n"
            + "\n".join(
                lines
            )
        )

    summary = f"""# Ankara Canonical Activity15 Spatial Permutation Importance

## Purpose

This diagnostic measures how much predictive ranking quality changes when one
canonical Activity15 feature is shuffled inside each validation fold.

The analysis uses the same fixed 5-km spatial folds and the same untuned Random
Forest and HistGradientBoosting configurations as the canonical ML evaluation.

Each model is fitted once per fold. A feature is then permuted only in the
validation data. The model is not retrained for each permutation.

## Configuration

- Canonical predictors: {len(CANONICAL_ML_FEATURE_COLUMNS)}
- Models: Random Forest and HistGradientBoosting
- Permutation repeats per feature/fold: {N_REPEATS}
- Primary importance statistic: pooled spatial OOF average-precision drop
- Supporting statistic: fold-level average-precision drop

{chr(10).join(sections)}

## Interpretation Policy

A positive AP drop means prediction quality deteriorated after the feature was
shuffled, which is evidence that the fitted model depended on information carried
by that feature.

A zero or negative drop does not prove that a feature is useless. Correlated or
substitutable predictors can mask one another, and the target contains only 46
positive cells.

Permutation importance is model-specific predictive dependence, not causal
importance.

The activity variables are mapped OSM urban-activity proxies. Importance should
not be interpreted as direct evidence of EV demand, trips, employment, traffic,
or commercial turnover.

This diagnostic should be read together with the earlier feature-family
incremental-value and seed-stability experiments.

## Outputs

- `data/processed/{IMPORTANCE_PATH.name}`
- `data/processed/{FOLD_DROP_PATH.name}`
- `docs/{PLOT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    importance: pd.DataFrame,
) -> None:
    """Print the top canonical permutation-importance features."""

    for model_name in MODEL_NAMES:
        print()

        print(
            MODEL_LABELS[
                model_name
            ]
        )

        print(
            importance.loc[
                importance[
                    "model"
                ]
                == model_name,
                [
                    "feature",
                    "feature_group",
                    "pooled_average_precision_drop",
                    "mean_fold_average_precision_drop",
                    "positive_fold_repeat_drop_fraction",
                ],
            ]
            .sort_values(
                "pooled_average_precision_drop",
                ascending=False,
                kind="stable",
            )
            .head(
                10
            )
            .to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )


def main() -> None:
    """Run canonical Activity15 spatial permutation importance."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara Canonical Activity15 Spatial Permutation Importance"
    )

    print(
        "="
        * 70
    )

    dataframe = load_inputs()

    (
        importance,
        fold_drops,
    ) = run_analysis(
        dataframe
    )

    save_outputs(
        importance,
        fold_drops,
    )

    create_plot(
        importance
    )

    create_summary(
        importance
    )

    print_results(
        importance
    )

    print(
        "="
        * 70
    )

    print(
        "Canonical Activity15 permutation importance completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
