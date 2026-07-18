from pathlib import Path

import matplotlib.pyplot as plt

from models.common.metrics import (
    METRIC_LABELS,
    higher_is_better,
)


def plot_model_comparison(
    validation_metrics,
    metric,
    endpoint_name,
    output_path,
):
    """
    Save a bar chart comparing:

    Extra Trees
    XGBoost
    CatBoost
    Weighted Ensemble
    """

    output_path = Path(output_path)

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    bars = axis.bar(
        validation_metrics["model"],
        validation_metrics[metric],
    )

    axis.set_title(
        f"{endpoint_name}: "
        f"Validation {METRIC_LABELS[metric]}"
    )

    axis.set_xlabel("Model")
    axis.set_ylabel(
        METRIC_LABELS[metric]
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    if higher_is_better(metric):
        direction = "Higher is better"
    else:
        direction = "Lower is better"

    axis.text(
        0.99,
        0.02,
        direction,
        transform=axis.transAxes,
        horizontalalignment="right",
    )

    for bar, value in zip(
        bars,
        validation_metrics[metric],
    ):
        axis.annotate(
            f"{value:.4f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 4),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_weight_search(
    weight_results,
    metric,
    endpoint_name,
    output_path,
):
    """
    Save a line graph showing ensemble performance
    for every XGBoost weight.
    """

    output_path = Path(output_path)

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        weight_results["xgboost_weight"],
        weight_results[metric],
        marker="o",
    )

    axis.set_title(
        f"{endpoint_name}: Ensemble Weight Search "
        f"({METRIC_LABELS[metric]})"
    )

    axis.set_xlabel(
        "XGBoost weight "
        "(CatBoost weight = 1 - weight)"
    )

    axis.set_ylabel(
        METRIC_LABELS[metric]
    )

    axis.set_xticks(
        weight_results["xgboost_weight"]
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.grid(alpha=0.3)

    if higher_is_better(metric):
        best_index = (
            weight_results[metric]
            .idxmax()
        )
    else:
        best_index = (
            weight_results[metric]
            .idxmin()
        )

    best_row = weight_results.loc[
        best_index
    ]

    axis.annotate(
        (
            f"Best: "
            f"XGB={best_row['xgboost_weight']:.2f}, "
            f"Cat={best_row['catboost_weight']:.2f}\n"
            f"{METRIC_LABELS[metric]}="
            f"{best_row[metric]:.4f}"
        ),

        xy=(
            best_row["xgboost_weight"],
            best_row[metric],
        ),

        xytext=(12, 12),

        textcoords="offset points",

        arrowprops={
            "arrowstyle": "->"
        },
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)