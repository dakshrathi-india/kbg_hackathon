import pandas as pd

from models.common.config import ENSEMBLE_WEIGHTS

from models.common.metrics import (
    calculate_metrics,
    get_best_index,
)


def search_ensemble_weights(
    y_valid,
    xgboost_predictions,
    catboost_predictions,
    task_type,
    primary_metric,
):
    """
    Search XGBoost weights from 0.05 to 0.95.

    CatBoost weight is:

        1 - XGBoost weight

    The best weight is selected using validation data only.
    """

    rows = {}
    prediction_by_weight = {}

    for xgb_weight in ENSEMBLE_WEIGHTS:

        catboost_weight = round(
            1.0 - xgb_weight,
            2,
        )

        ensemble_predictions = (
            xgb_weight
            * xgboost_predictions
            +
            catboost_weight
            * catboost_predictions
        )

        metrics = calculate_metrics(
            y_true=y_valid,
            predictions=ensemble_predictions,
            task_type=task_type,
        )

        rows[xgb_weight] = {
            "xgboost_weight": xgb_weight,
            "catboost_weight": catboost_weight,
            **metrics,
        }

        prediction_by_weight[
            xgb_weight
        ] = ensemble_predictions

    weight_results = pd.DataFrame(
        rows.values()
    )

    best_index = get_best_index(
        weight_results[primary_metric],
        primary_metric,
    )

    best_row = (
        weight_results
        .loc[best_index]
        .to_dict()
    )

    best_xgb_weight = float(
        best_row["xgboost_weight"]
    )

    best_predictions = prediction_by_weight[
        best_xgb_weight
    ]

    return (
        weight_results,
        best_row,
        best_predictions,
    )