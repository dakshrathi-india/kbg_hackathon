import numpy as np
import pandas as pd

from scipy.optimize import minimize

from models.common.metrics import (
    calculate_metrics,
    higher_is_better,
)


# ============================================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================

def search_ensemble_weights(
    y_valid,
    predictions_by_model,
    task_type,
    primary_metric,
):
    """
    Optimize ensemble weights for any number of models.

    Parameters
    ----------
    y_valid:
        True validation targets.

    predictions_by_model:
        Dictionary:

        {
            "Extra Trees": predictions,
            "XGBoost": predictions,
            ...
        }

    task_type:
        "regression" or "classification"

    primary_metric:
        Metric used to select weights.

        Regression:
            rmse
            r2

        Classification:
            roc_auc
            auprc


    Weight constraints
    ------------------
    Every weight:

        0 <= weight <= 1

    All weights:

        sum(weights) = 1


    Returns
    -------
    weight_results

        DataFrame containing the optimized
        weights and validation metrics.


    best_row

        Dictionary containing optimized
        weights and metrics.


    best_predictions

        Final ensemble predictions.
    """


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if not predictions_by_model:

        raise ValueError(
            "predictions_by_model "
            "cannot be empty."
        )


    y_valid = np.asarray(
        y_valid,
        dtype=float,
    )


    model_names = list(
        predictions_by_model.keys()
    )


    number_of_models = len(
        model_names
    )


    print(
        "\nOptimizing ensemble using:"
    )


    for model_name in model_names:

        print(
            f"  - {model_name}"
        )


    # ========================================================
    # CREATE PREDICTION MATRIX
    #
    # Shape:
    #
    # (number_of_samples, number_of_models)
    # ========================================================

    prediction_arrays = []


    for model_name in model_names:

        predictions = np.asarray(

            predictions_by_model[
                model_name
            ],

            dtype=float,
        )


        if len(predictions) != len(
            y_valid
        ):

            raise ValueError(

                f"Prediction length mismatch "
                f"for model: {model_name}"
            )


        prediction_arrays.append(
            predictions
        )


    prediction_matrix = np.column_stack(

        prediction_arrays
    )


    # ========================================================
    # INITIAL WEIGHTS
    #
    # Start with equal weights.
    # ========================================================

    initial_weights = np.full(

        number_of_models,

        1.0
        /
        number_of_models,
    )


    # ========================================================
    # WEIGHT CONSTRAINTS
    # ========================================================

    bounds = [

        (
            0.0,
            1.0,
        )

        for _ in range(
            number_of_models
        )
    ]


    constraints = [

        {

            "type": "eq",

            "fun": lambda weights: (

                np.sum(
                    weights
                )

                - 1.0
            ),
        }
    ]


    # ========================================================
    # OBJECTIVE FUNCTION
    # ========================================================

    def objective(
        weights
    ):

        ensemble_predictions = (

            prediction_matrix

            @

            weights
        )


        metrics = calculate_metrics(

            y_true=y_valid,

            predictions=(
                ensemble_predictions
            ),

            task_type=(
                task_type
            ),
        )


        metric_value = float(

            metrics[
                primary_metric
            ]
        )


        # scipy.optimize MINIMIZES.
        #
        # RMSE:
        #     lower is better
        #     return directly
        #
        # R² / ROC-AUC / AUPRC:
        #     higher is better
        #     return negative value

        if higher_is_better(
            primary_metric
        ):

            return (
                -metric_value
            )


        return metric_value


    # ========================================================
    # OPTIMIZE WEIGHTS
    # ========================================================

    optimization_result = minimize(

        objective,

        initial_weights,

        method="SLSQP",

        bounds=bounds,

        constraints=constraints,

        options={

            "maxiter": 1000,

            "ftol": 1e-10,
        },
    )


    # ========================================================
    # CHECK OPTIMIZATION
    # ========================================================

    if not optimization_result.success:

        print(
            "\nWARNING:"
            "\nEnsemble optimization did not "
            "fully converge."
        )

        print(
            optimization_result.message
        )


    # ========================================================
    # GET BEST WEIGHTS
    # ========================================================

    best_weights = np.asarray(

        optimization_result.x,

        dtype=float,
    )


    # Remove tiny floating-point values.

    best_weights[

        np.abs(
            best_weights
        )

        < 1e-8

    ] = 0.0


    # Normalize again for numerical safety.

    weight_sum = np.sum(
        best_weights
    )


    if weight_sum <= 0:

        raise ValueError(
            "Optimized ensemble weights "
            "sum to zero."
        )


    best_weights = (

        best_weights

        /

        weight_sum
    )


    # ========================================================
    # BEST ENSEMBLE PREDICTIONS
    # ========================================================

    best_predictions = (

        prediction_matrix

        @

        best_weights
    )


    # ========================================================
    # CALCULATE BEST METRICS
    # ========================================================

    best_metrics = calculate_metrics(

        y_true=y_valid,

        predictions=(
            best_predictions
        ),

        task_type=(
            task_type
        ),
    )


    # ========================================================
    # CREATE RESULT ROW
    # ========================================================

    best_row = {}


    for (
        model_name,
        weight,
    ) in zip(

        model_names,

        best_weights,
    ):

        weight_name = (

            model_name
            .lower()
            .replace(
                " ",
                "_"
            )
            .replace(
                "-",
                "_"
            )

            + "_weight"
        )


        best_row[
            weight_name
        ] = float(
            weight
        )


    best_row.update(

        {

            metric_name: float(
                metric_value
            )

            for (
                metric_name,
                metric_value,
            )

            in best_metrics.items()
        }
    )


    # ========================================================
    # RESULTS DATAFRAME
    #
    # Since this is optimization rather than brute-force
    # grid search, this contains the final optimized solution.
    # ========================================================

    weight_results = pd.DataFrame(

        [
            best_row
        ]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\nOptimized ensemble weights:"
    )


    for (
        model_name,
        weight,
    ) in zip(

        model_names,

        best_weights,
    ):

        print(

            f"  {model_name}: "
            f"{weight:.4f}"
        )


    print(
        "\nOptimized ensemble metrics:"
    )


    for (
        metric_name,
        metric_value,
    ) in best_metrics.items():

        print(

            f"  {metric_name}: "
            f"{metric_value:.5f}"
        )


    # ========================================================
    # RETURN
    # ========================================================

    return (

        weight_results,

        best_row,

        best_predictions,
    )