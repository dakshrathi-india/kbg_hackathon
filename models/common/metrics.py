import numpy as np
from scipy.stats import spearmanr

from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "auprc": "AUPRC",
    "rmse": "RMSE",
    "r2": "R²",
}


def calculate_metrics(
    y_true,
    predictions,
    task_type,
):
    """
    Calculate validation metrics.

    Classification:
        ROC-AUC
        AUPRC

    Regression:
        RMSE
        R²
    """

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    if len(y_true) != len(predictions):
        raise ValueError(
            "y_true and predictions must have the same length."
        )

    if task_type == "classification":

        if np.unique(y_true).size < 2:
            raise ValueError(
                "Validation data contains only one class. "
                "ROC-AUC cannot be calculated."
            )

        return {
            "roc_auc": float(
                roc_auc_score(
                    y_true,
                    predictions,
                )
            ),

            "auprc": float(
                average_precision_score(
                    y_true,
                    predictions,
                )
            ),
        }

    if task_type == "regression":

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        )

        return {
            "rmse": float(rmse),

            "r2": float(
                r2_score(
                    y_true,
                    predictions,
                )
            ),
        }

    raise ValueError(
        "task_type must be classification or regression."
    )


def higher_is_better(metric):
    """
    Tell whether a larger value means better performance.
    """

    if metric in {"roc_auc", "auprc", "r2"}:
        return True

    if metric == "rmse":
        return False

    raise ValueError(
        f"Unknown metric: {metric}"
    )


def get_best_index(
    metric_values,
    metric,
):
    """
    Return the DataFrame index containing the best score.
    """

    if higher_is_better(metric):
        return metric_values.idxmax()

    return metric_values.idxmin()