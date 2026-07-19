import json
from pathlib import Path

import joblib
import numpy as np

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)

from lightgbm import (
    LGBMClassifier,
    LGBMRegressor,
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
    get_task_config,
)

from models.common.ensemble import (
    search_ensemble_weights,
)

from models.common.metrics import (
    calculate_metrics,
)


# ============================================================
# SETTINGS
# ============================================================

TASKS = [
    "a",
    "d",
    "m",
    "e",
    "t",
    "s",
]

SEEDS = [
    42,
    52,
    62,
]

ARTIFACT_ROOT = Path(
    "model_artifacts"
)


# ============================================================
# E = CLEARANCE
# OPTUNA-TUNED PARAMETERS
# ============================================================

E_TUNED_PARAMS = {

    "XGBoost": {

        "n_estimators": 500,

        "max_depth": 5,

        "learning_rate":
            0.01214628620246294,

        "min_child_weight": 4,

        "subsample":
            0.7080440328077426,

        "colsample_bytree":
            0.6071572227762091,

        "reg_alpha":
            0.00775097079614023,

        "reg_lambda":
            0.22011687418907247,

        "gamma":
            0.38216688465603343,
    },


    "CatBoost": {

        "iterations": 1200,

        "depth": 8,

        "learning_rate":
            0.013614726143703801,

        "l2_leaf_reg":
            12.655008234221212,

        "random_strength":
            0.8740950779496219,

        "bagging_temperature":
            1.2050325615904873,
    },


    "LightGBM": {

        "n_estimators": 600,

        "num_leaves": 21,

        "max_depth": 6,

        "learning_rate":
            0.025116102816601968,

        "min_child_samples": 14,

        "subsample":
            0.8395505804205186,

        "colsample_bytree":
            0.5326656872530611,

        "reg_alpha":
            0.06454844300259355,

        "reg_lambda":
            0.1253011537233605,
    },
}


# ============================================================
# CLASS IMBALANCE
# ============================================================

def calculate_scale_pos_weight(
    y_train,
):

    y_train = np.asarray(
        y_train
    ).astype(int)


    positive_count = int(

        np.sum(
            y_train == 1
        )
    )


    negative_count = int(

        np.sum(
            y_train == 0
        )
    )


    if (
        positive_count == 0
        or negative_count == 0
    ):

        raise ValueError(

            "Classification training data "
            "must contain both classes."
        )


    return (

        negative_count
        /

        positive_count
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(

    model_name,

    task,

    task_type,

    seed,

    y_train,

):


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if task_type == "classification":


        scale_pos_weight = (

            calculate_scale_pos_weight(
                y_train
            )
        )


        # ----------------------------------------------------
        # EXTRA TREES
        # ----------------------------------------------------

        if model_name == "Extra Trees":


            return (

                ExtraTreesClassifier(

                    n_estimators=500,

                    class_weight="balanced",

                    random_state=seed,

                    n_jobs=-1,
                )
            )


        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        if model_name == "XGBoost":


            return (

                XGBClassifier(

                    n_estimators=600,

                    max_depth=6,

                    learning_rate=0.05,

                    min_child_weight=2,

                    subsample=0.85,

                    colsample_bytree=0.80,

                    reg_alpha=0.05,

                    reg_lambda=1.50,

                    objective=(
                        "binary:logistic"
                    ),

                    eval_metric="logloss",

                    scale_pos_weight=(
                        scale_pos_weight
                    ),

                    tree_method="hist",

                    device="cuda",

                    random_state=seed,

                    n_jobs=-1,
                )
            )


        # ----------------------------------------------------
        # CATBOOST
        # ----------------------------------------------------

        if model_name == "CatBoost":


            return (

                CatBoostClassifier(

                    iterations=600,

                    depth=7,

                    learning_rate=0.05,

                    loss_function="Logloss",

                    auto_class_weights=(
                        "Balanced"
                    ),

                    task_type="GPU",

                    random_seed=seed,

                    verbose=False,

                    allow_writing_files=False,
                )
            )


        # ----------------------------------------------------
        # LIGHTGBM
        # ----------------------------------------------------

        if model_name == "LightGBM":


            return (

                LGBMClassifier(

                    n_estimators=600,

                    num_leaves=31,

                    max_depth=-1,

                    learning_rate=0.05,

                    subsample=0.85,

                    colsample_bytree=0.80,

                    reg_alpha=0.05,

                    reg_lambda=1.50,

                    class_weight="balanced",

                    random_state=seed,

                    n_jobs=-1,

                    verbosity=-1,
                )
            )


    # ========================================================
    # REGRESSION
    # ========================================================

    if task_type == "regression":


        # ----------------------------------------------------
        # EXTRA TREES
        # ----------------------------------------------------

        if model_name == "Extra Trees":


            return (

                ExtraTreesRegressor(

                    n_estimators=500,

                    random_state=seed,

                    n_jobs=-1,
                )
            )


        # ====================================================
        # E = CLEARANCE
        #
        # USE OPTUNA-TUNED PARAMETERS
        # ====================================================

        if task == "e":


            # ------------------------------------------------
            # TUNED XGBOOST
            # ------------------------------------------------

            if model_name == "XGBoost":


                return (

                    XGBRegressor(

                        **E_TUNED_PARAMS[
                            "XGBoost"
                        ],

                        objective=(
                            "reg:squarederror"
                        ),

                        eval_metric="rmse",

                        tree_method="hist",

                        device="cuda",

                        random_state=seed,

                        n_jobs=-1,
                    )
                )


            # ------------------------------------------------
            # TUNED CATBOOST
            # ------------------------------------------------

            if model_name == "CatBoost":


                return (

                    CatBoostRegressor(

                        **E_TUNED_PARAMS[
                            "CatBoost"
                        ],

                        loss_function="RMSE",

                        task_type="GPU",

                        random_seed=seed,

                        verbose=False,

                        allow_writing_files=False,
                    )
                )


            # ------------------------------------------------
            # TUNED LIGHTGBM
            # ------------------------------------------------

            if model_name == "LightGBM":


                return (

                    LGBMRegressor(

                        **E_TUNED_PARAMS[
                            "LightGBM"
                        ],

                        objective="regression",

                        random_state=seed,

                        n_jobs=-1,

                        verbosity=-1,
                    )
                )


        # ====================================================
        # STANDARD REGRESSION MODELS
        #
        # A AND S
        # ====================================================

        if model_name == "XGBoost":


            return (

                XGBRegressor(

                    n_estimators=600,

                    max_depth=6,

                    learning_rate=0.05,

                    min_child_weight=2,

                    subsample=0.85,

                    colsample_bytree=0.80,

                    reg_alpha=0.05,

                    reg_lambda=1.50,

                    objective=(
                        "reg:squarederror"
                    ),

                    eval_metric="rmse",

                    tree_method="hist",

                    device="cuda",

                    random_state=seed,

                    n_jobs=-1,
                )
            )


        if model_name == "CatBoost":


            return (

                CatBoostRegressor(

                    iterations=600,

                    depth=7,

                    learning_rate=0.05,

                    loss_function="RMSE",

                    task_type="GPU",

                    random_seed=seed,

                    verbose=False,

                    allow_writing_files=False,
                )
            )


        if model_name == "LightGBM":


            return (

                LGBMRegressor(

                    n_estimators=600,

                    num_leaves=31,

                    max_depth=-1,

                    learning_rate=0.05,

                    subsample=0.85,

                    colsample_bytree=0.80,

                    reg_alpha=0.05,

                    reg_lambda=1.50,

                    objective="regression",

                    random_state=seed,

                    n_jobs=-1,

                    verbosity=-1,
                )
            )


    raise ValueError(

        f"Could not create model: "

        f"{model_name}, "

        f"task={task}, "

        f"task_type={task_type}"
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(

    model,

    X,

    task_type,

):


    if task_type == "classification":


        return (

            model
            .predict_proba(
                X
            )[:, 1]
        )


    return (

        model.predict(
            X
        )
    )


# ============================================================
# FILE NAME
# ============================================================

def get_model_filename(

    model_name,

    seed,

):


    model_key = (

        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    return (

        f"{model_key}"

        f"_seed_{seed}"

        f".joblib"
    )


# ============================================================
# TRAIN ONE ENDPOINT
# ============================================================

def train_endpoint(
    task,
):


    # ========================================================
    # CONFIG
    # ========================================================

    config = (

        get_task_config(
            task
        )
    )


    task_type = (

        config[
            "task_type"
        ]
    )


    print(
        "\n"
        + "=" * 70
    )


    print(

        f"FINAL TRAINING: "

        f"{task.upper()} "

        f"- {config['name']}"
    )


    print(
        "=" * 70
    )


    # ========================================================
    # ARTIFACT FOLDER
    # ========================================================

    artifact_folder = (

        ARTIFACT_ROOT
        /
        task
    )


    artifact_folder.mkdir(

        parents=True,

        exist_ok=True,
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    data = (

        load_data(
            task
        )
    )


    X_train = (

        data[
            "X_train"
        ]
    )


    y_train = (

        data[
            "y_train"
        ]
    )


    X_valid = (

        data[
            "X_valid"
        ]
    )


    y_valid = (

        data[
            "y_valid"
        ]
    )


    if task_type == "classification":


        y_train = (

            y_train.astype(
                int
            )
        )


        y_valid = (

            y_valid.astype(
                int
            )
        )


    print(

        f"Train shape: "

        f"{X_train.shape}"
    )


    print(

        f"Validation shape: "

        f"{X_valid.shape}"
    )


    # ========================================================
    # IMPUTER
    # ========================================================

    imputer = (

        SimpleImputer(

            strategy="median",

            keep_empty_features=True,
        )
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

        artifact_folder
        /
        "imputer.joblib",
    )


    # ========================================================
    # MODEL FAMILIES
    # ========================================================

    model_names = [

        "Extra Trees",

        "XGBoost",

        "CatBoost",

        "LightGBM",
    ]


    # ========================================================
    # STORE SEED PREDICTIONS
    # ========================================================

    seed_predictions = {

        model_name: []

        for model_name

        in model_names
    }


    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for model_name in model_names:


        print(

            f"\n--- "

            f"{model_name} "

            f"---"
        )


        for seed in SEEDS:


            print(

                f"Training "

                f"{model_name} "

                f"with seed "

                f"{seed}..."
            )


            # ------------------------------------------------
            # CREATE MODEL
            # ------------------------------------------------

            model = (

                create_model(

                    model_name=(
                        model_name
                    ),

                    task=(
                        task
                    ),

                    task_type=(
                        task_type
                    ),

                    seed=(
                        seed
                    ),

                    y_train=(
                        y_train
                    ),
                )
            )


            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            model.fit(

                X_train_ready,

                y_train,
            )


            # ------------------------------------------------
            # VALIDATION PREDICTIONS
            # ------------------------------------------------

            predictions = (

                generate_predictions(

                    model=(
                        model
                    ),

                    X=(
                        X_valid_ready
                    ),

                    task_type=(
                        task_type
                    ),
                )
            )


            predictions = (

                np.asarray(
                    predictions
                )
            )


            seed_predictions[
                model_name
            ].append(

                predictions
            )


            # ------------------------------------------------
            # SAVE MODEL
            # ------------------------------------------------

            filename = (

                get_model_filename(

                    model_name=(
                        model_name
                    ),

                    seed=(
                        seed
                    ),
                )
            )


            joblib.dump(

                model,

                artifact_folder
                /
                filename,
            )


            print(

                f"Saved: "

                f"{filename}"
            )


    # ========================================================
    # AVERAGE THREE SEEDS
    # ========================================================

    averaged_predictions = {}


    print(

        "\n"
        + "-" * 70
    )


    print(

        "AVERAGED SEED RESULTS"
    )


    print(

        "-" * 70
    )


    for model_name in model_names:


        prediction_matrix = (

            np.vstack(

                seed_predictions[
                    model_name
                ]
            )
        )


        average_prediction = (

            np.mean(

                prediction_matrix,

                axis=0,
            )
        )


        averaged_predictions[
            model_name
        ] = (

            average_prediction
        )


        metrics = (

            calculate_metrics(

                y_true=(
                    y_valid
                ),

                predictions=(
                    average_prediction
                ),

                task_type=(
                    task_type
                ),
            )
        )


        metric_text = ", ".join(

            f"{key}="
            f"{value:.5f}"

            for (
                key,
                value,
            )

            in metrics.items()
        )


        print(

            f"{model_name}: "

            f"{metric_text}"
        )


    # ========================================================
    # OPTIMIZE FOUR FAMILY-LEVEL WEIGHTS
    # ========================================================

    print(

        "\nOptimizing final "

        "ensemble weights..."
    )


    (

        weight_results,

        best_weight_row,

        best_ensemble_predictions,

    ) = search_ensemble_weights(


        y_valid=(
            y_valid
        ),


        predictions_by_model=(

            averaged_predictions
        ),


        task_type=(
            task_type
        ),


        primary_metric=(

            config[
                "primary_metric"
            ]
        ),
    )


    # ========================================================
    # FINAL ENSEMBLE METRICS
    # ========================================================

    final_metrics = (

        calculate_metrics(

            y_true=(
                y_valid
            ),

            predictions=(
                best_ensemble_predictions
            ),

            task_type=(
                task_type
            ),
        )
    )


    # ========================================================
    # EXTRACT WEIGHTS
    # ========================================================

    final_weights = {}


    for model_name in model_names:


        weight_key = (

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


        final_weights[
            model_name
        ] = float(

            best_weight_row[
                weight_key
            ]
        )


    # ========================================================
    # CREATE ENSEMBLE CONFIG
    # ========================================================

    ensemble_config = {


        "task": (
            task
        ),


        "endpoint": (

            config[
                "name"
            ]
        ),


        "dataset": (

            config[
                "dataset"
            ]
        ),


        "task_type": (

            task_type
        ),


        "seeds": (

            SEEDS
        ),


        "seed_aggregation": (

            "mean"
        ),


        "model_families": (

            model_names
        ),


        "ensemble_weights": (

            final_weights
        ),


        "selected_using": (

            config[
                "primary_metric"
            ]
        ),


        "validation_metrics": {

            metric_name: float(
                metric_value
            )

            for (
                metric_name,
                metric_value,
            )

            in final_metrics.items()
        },


        "target_transformation": (

            "none"
        ),


        "e_uses_tuned_parameters": (

            task == "e"
        ),
    }


    # ========================================================
    # SAVE CONFIG
    # ========================================================

    with open(

        artifact_folder
        /
        "ensemble_config.json",

        "w",

        encoding="utf-8",

    ) as file:


        json.dump(

            ensemble_config,

            file,

            indent=4,
        )


    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print(

        "\n"
        + "=" * 70
    )


    print(

        f"FINAL RESULT: "

        f"{task.upper()}"
    )


    print(

        "=" * 70
    )


    print(

        "\nEnsemble weights:"
    )


    for (

        model_name,
        weight,

    ) in final_weights.items():


        print(

            f"  {model_name}: "

            f"{weight:.4f}"
        )


    print(

        "\nValidation metrics:"
    )


    for (

        metric_name,
        metric_value,

    ) in final_metrics.items():


        print(

            f"  {metric_name}: "

            f"{metric_value:.5f}"
        )


    print(

        f"\nArtifacts saved to: "

        f"{artifact_folder}"
    )


    return (

        ensemble_config
    )


# ============================================================
# MAIN
# ============================================================

def main():


    ARTIFACT_ROOT.mkdir(

        parents=True,

        exist_ok=True,
    )


    summaries = {}


    for task in TASKS:


        try:


            summaries[
                task
            ] = (

                train_endpoint(
                    task
                )
            )


        except Exception as error:


            print(

                f"\nERROR training "

                f"{task.upper()}: "

                f"{error}"
            )


            raise


    # ========================================================
    # SAVE GLOBAL SUMMARY
    # ========================================================

    summary_path = (

        ARTIFACT_ROOT
        /
        "training_summary.json"
    )


    with open(

        summary_path,

        "w",

        encoding="utf-8",

    ) as file:


        json.dump(

            summaries,

            file,

            indent=4,
        )


    print(

        "\n"
        + "=" * 70
    )


    print(

        "ALL FINAL MODELS TRAINED"
    )


    print(

        "=" * 70
    )


    for task in TASKS:


        metrics = (

            summaries[
                task
            ][
                "validation_metrics"
            ]
        )


        metric_text = ", ".join(

            f"{key}="
            f"{value:.5f}"

            for (
                key,
                value,
            )

            in metrics.items()
        )


        print(

            f"{task.upper()}: "

            f"{metric_text}"
        )


    print(

        f"\nAll artifacts saved in: "

        f"{ARTIFACT_ROOT}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()