import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

from sklearn.impute import SimpleImputer

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)


from data_pipeline.data_loader import load_data

from models.common.config import (
    RANDOM_STATE,
    RESULTS_ROOT,
    SAVED_MODELS_ROOT,
    get_task_config,
)

from models.common.ensemble import (
    search_ensemble_weights,
)

from models.common.metrics import (
    calculate_metrics,
    get_best_index,
)

from models.common.plotting import (
    plot_model_comparison,
    plot_weight_search,
)


def create_output_folders(task):
    """
    Automatically create:

        results/<task>
        saved_models/<task>
    """

    results_folder = (
        RESULTS_ROOT / task
    )

    saved_models_folder = (
        SAVED_MODELS_ROOT / task
    )

    results_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_models_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        results_folder,
        saved_models_folder,
    )


def calculate_scale_pos_weight(y_train):
    """
    Calculate class-imbalance weight for XGBoost.

    scale_pos_weight =
        number of negative samples
        /
        number of positive samples
    """

    y_train = np.asarray(
        y_train
    ).astype(int)

    positive_count = int(
        np.sum(y_train == 1)
    )

    negative_count = int(
        np.sum(y_train == 0)
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        raise ValueError(
            "Training data must contain "
            "both class 0 and class 1."
        )

    return (
        negative_count
        /
        positive_count
    )


def create_models(
    task_type,
    y_train,
    use_gpu,
):
    """
    Create Extra Trees, XGBoost and CatBoost models.
    """

    if use_gpu:
        xgboost_device = "cuda"
        catboost_task_type = "GPU"
    else:
        xgboost_device = "cpu"
        catboost_task_type = "CPU"

    if task_type == "classification":

        scale_pos_weight = (
            calculate_scale_pos_weight(
                y_train
            )
        )

        models = {
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

            "XGBoost": XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=1.50,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=(
                    scale_pos_weight
                ),
                tree_method="hist",
                device=xgboost_device,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

            "CatBoost": CatBoostClassifier(
                iterations=600,
                depth=7,
                learning_rate=0.05,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                task_type=catboost_task_type,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        }

        return models

    if task_type == "regression":

        models = {
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

            "XGBoost": XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=1.50,
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                device=xgboost_device,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

            "CatBoost": CatBoostRegressor(
                iterations=600,
                depth=7,
                learning_rate=0.05,
                loss_function="RMSE",
                task_type=catboost_task_type,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        }

        return models

    raise ValueError(
        "task_type must be classification or regression."
    )


def generate_predictions(
    model,
    X_valid,
    task_type,
):
    """
    Classification:
        Return probability of class 1.

    Regression:
        Return predicted numerical value.
    """

    if task_type == "classification":
        return model.predict_proba(
            X_valid
        )[:, 1]

    return model.predict(
        X_valid
    )


def run_experiment(
    task,
    use_gpu=False,
):
    """
    Run the complete validation experiment
    for one ADMET endpoint.

    The test set is not used.
    """

    config = get_task_config(task)

    task = config["task"]

    print("\n" + "=" * 70)

    print(
        f"{config['name']} "
        f"({config['dataset']}) "
        f"- {config['task_type']}"
    )

    print("=" * 70)

    (
        results_folder,
        saved_models_folder,
    ) = create_output_folders(task)

    # Load all saved arrays.
    data = load_data(task)

    # Use only train and validation for now.
    X_train = data["X_train"]
    y_train = data["y_train"]

    X_valid = data["X_valid"]
    y_valid = data["y_valid"]

    if (
        config["task_type"]
        == "classification"
    ):
        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_valid shape: {X_valid.shape}"
    )

    # Learn missing-value replacements
    # using training data only.
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X_train_ready = (
        imputer.fit_transform(
            X_train
        )
    )

    X_valid_ready = (
        imputer.transform(
            X_valid
        )
    )

    joblib.dump(
        imputer,
        saved_models_folder
        / "imputer.joblib",
    )

    models = create_models(
        task_type=config["task_type"],
        y_train=y_train,
        use_gpu=use_gpu,
    )

    validation_rows = []

    validation_predictions = {}

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train_ready,
            y_train,
        )

        predictions = generate_predictions(
            model=model,
            X_valid=X_valid_ready,
            task_type=config["task_type"],
        )

        metrics = calculate_metrics(
            y_true=y_valid,
            predictions=predictions,
            task_type=config["task_type"],
        )

        validation_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        validation_predictions[
            model_name
        ] = predictions

        model_filename = (
            model_name
            .lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            saved_models_folder
            / model_filename,
        )

        metric_text = ", ".join(
            (
                f"{metric_name}="
                f"{metric_value:.5f}"
            )
            for (
                metric_name,
                metric_value,
            ) in metrics.items()
        )

        print(
            f"Validation: {metric_text}"
        )

    (
        weight_results,
        best_weight_row,
        best_ensemble_predictions,
    ) = search_ensemble_weights(
        y_valid=y_valid,

        xgboost_predictions=(
            validation_predictions[
                "XGBoost"
            ]
        ),

        catboost_predictions=(
            validation_predictions[
                "CatBoost"
            ]
        ),

        task_type=config["task_type"],

        primary_metric=(
            config["primary_metric"]
        ),
    )

    best_ensemble_metrics = (
        calculate_metrics(
            y_true=y_valid,

            predictions=(
                best_ensemble_predictions
            ),

            task_type=(
                config["task_type"]
            ),
        )
    )

    validation_rows.append(
        {
            "model": "Weighted Ensemble",
            **best_ensemble_metrics,
        }
    )

    validation_predictions[
        "Weighted Ensemble"
    ] = best_ensemble_predictions

    validation_metrics = pd.DataFrame(
        validation_rows
    )

    validation_metrics.to_csv(
        results_folder
        / "validation_metrics.csv",
        index=False,
    )

    weight_results.to_csv(
        results_folder
        / "ensemble_weight_results.csv",
        index=False,
    )

    # Save predictions because these can later
    # be combined with GNN predictions.
    predictions_table = pd.DataFrame(
        {
            "y_true": y_valid,

            "extra_trees_prediction": (
                validation_predictions[
                    "Extra Trees"
                ]
            ),

            "xgboost_prediction": (
                validation_predictions[
                    "XGBoost"
                ]
            ),

            "catboost_prediction": (
                validation_predictions[
                    "CatBoost"
                ]
            ),

            "weighted_ensemble_prediction": (
                validation_predictions[
                    "Weighted Ensemble"
                ]
            ),
        }
    )

    predictions_table.to_csv(
        results_folder
        / "validation_predictions.csv",
        index=False,
    )

    # Save graphs for both required metrics.
    for metric in config["metrics"]:

        plot_model_comparison(
            validation_metrics=(
                validation_metrics
            ),

            metric=metric,

            endpoint_name=(
                config["name"]
            ),

            output_path=(
                results_folder
                /
                f"{metric}_model_comparison.png"
            ),
        )

        plot_weight_search(
            weight_results=(
                weight_results
            ),

            metric=metric,

            endpoint_name=(
                config["name"]
            ),

            output_path=(
                results_folder
                /
                f"{metric}_ensemble_weight_search.png"
            ),
        )

    # Select the best of:
    # Extra Trees, XGBoost, CatBoost,
    # and weighted ensemble.
    best_solution_index = (
        get_best_index(
            validation_metrics[
                config["primary_metric"]
            ],

            config["primary_metric"],
        )
    )

    best_solution_row = (
        validation_metrics
        .loc[best_solution_index]
        .to_dict()
    )

    ensemble_config = {
        "xgboost_weight": float(
            best_weight_row[
                "xgboost_weight"
            ]
        ),

        "catboost_weight": float(
            best_weight_row[
                "catboost_weight"
            ]
        ),

        "selected_using": (
            config["primary_metric"]
        ),

        "validation_metrics": {
            metric: float(
                best_weight_row[metric]
            )
            for metric in config["metrics"]
        },
    }

    with open(
        saved_models_folder
        / "ensemble_config.json",

        "w",

        encoding="utf-8",
    ) as file:

        json.dump(
            ensemble_config,
            file,
            indent=4,
        )

    validation_summary = {
        "task": task,

        "endpoint": config["name"],

        "dataset": config["dataset"],

        "task_type": (
            config["task_type"]
        ),

        "primary_metric": (
            config["primary_metric"]
        ),

        "best_validation_solution": (
            best_solution_row["model"]
        ),

        "best_validation_metrics": {
            metric: float(
                best_solution_row[metric]
            )
            for metric in config["metrics"]
        },

        "best_xgboost_catboost_ensemble": (
            ensemble_config
        ),

        "gpu_requested": use_gpu,

        "test_set_used": False,
    }

    with open(
        results_folder
        / "validation_summary.json",

        "w",

        encoding="utf-8",
    ) as file:

        json.dump(
            validation_summary,
            file,
            indent=4,
        )

    print("\nValidation comparison")

    print(
        validation_metrics.to_string(
            index=False
        )
    )

    print(
        "\nBest ensemble weights: "
        f"XGBoost="
        f"{ensemble_config['xgboost_weight']:.2f}, "
        f"CatBoost="
        f"{ensemble_config['catboost_weight']:.2f}"
    )

    print(
        "Best validation solution: "
        f"{validation_summary['best_validation_solution']}"
    )

    print(
        f"Results saved in: "
        f"{results_folder}"
    )

    print(
        f"Models saved in: "
        f"{saved_models_folder}"
    )

    return validation_summary