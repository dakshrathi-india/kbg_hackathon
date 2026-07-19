import json

import joblib
import numpy as np
import pandas as pd

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
)


# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders(task):

    results_folder = RESULTS_ROOT / task

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


# ============================================================
# 2. CLASS IMBALANCE WEIGHT
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
            "Training data must contain "
            "both class 0 and class 1."
        )

    return (
        negative_count
        /
        positive_count
    )


# ============================================================
# 3. CREATE MODELS
# ============================================================

def create_models(
    task,
    task_type,
    y_train,
    use_gpu,
):

    """
    Classification endpoints:

        Extra Trees
        XGBoost
        CatBoost
        LightGBM


    Normal regression endpoints A and S:

        Extra Trees
        XGBoost
        CatBoost
        LightGBM


    E = Clearance:

        Extra Trees

        Original XGBoost
        Pseudo-Huber XGBoost

        Original CatBoost
        Huber CatBoost

        Original LightGBM
        Huber LightGBM
    """


    # ========================================================
    # GPU SETTINGS
    # ========================================================

    if use_gpu:

        xgboost_device = "cuda"

        catboost_task_type = "GPU"

    else:

        xgboost_device = "cpu"

        catboost_task_type = "CPU"


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if task_type == "classification":

        scale_pos_weight = (
            calculate_scale_pos_weight(
                y_train
            )
        )


        models = {


            # ------------------------------------------------
            # EXTRA TREES
            # ------------------------------------------------

            "Extra Trees": (

                ExtraTreesClassifier(

                    n_estimators=500,

                    class_weight="balanced",

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,
                )
            ),


            # ------------------------------------------------
            # XGBOOST
            # ------------------------------------------------

            "XGBoost": (

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

                    eval_metric=(
                        "logloss"
                    ),

                    scale_pos_weight=(
                        scale_pos_weight
                    ),

                    tree_method="hist",

                    device=(
                        xgboost_device
                    ),

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,
                )
            ),


            # ------------------------------------------------
            # CATBOOST
            # ------------------------------------------------

            "CatBoost": (

                CatBoostClassifier(

                    iterations=600,

                    depth=7,

                    learning_rate=0.05,

                    loss_function=(
                        "Logloss"
                    ),

                    auto_class_weights=(
                        "Balanced"
                    ),

                    task_type=(
                        catboost_task_type
                    ),

                    random_seed=(
                        RANDOM_STATE
                    ),

                    verbose=False,

                    allow_writing_files=False,
                )
            ),


            # ------------------------------------------------
            # LIGHTGBM
            # ------------------------------------------------

            "LightGBM": (

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

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,

                    verbosity=-1,
                )
            ),
        }


        return models


    # ========================================================
    # REGRESSION
    # ========================================================

    if task_type == "regression":


        # ====================================================
        # STANDARD MODELS
        # ====================================================

        models = {


            # ------------------------------------------------
            # EXTRA TREES
            # ------------------------------------------------

            "Extra Trees": (

                ExtraTreesRegressor(

                    n_estimators=500,

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,
                )
            ),


            # ------------------------------------------------
            # ORIGINAL XGBOOST
            # ------------------------------------------------

            "XGBoost": (

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

                    device=(
                        xgboost_device
                    ),

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,
                )
            ),


            # ------------------------------------------------
            # ORIGINAL CATBOOST
            # ------------------------------------------------

            "CatBoost": (

                CatBoostRegressor(

                    iterations=600,

                    depth=7,

                    learning_rate=0.05,

                    loss_function="RMSE",

                    task_type=(
                        catboost_task_type
                    ),

                    random_seed=(
                        RANDOM_STATE
                    ),

                    verbose=False,

                    allow_writing_files=False,
                )
            ),


            # ------------------------------------------------
            # ORIGINAL LIGHTGBM
            # ------------------------------------------------

            "LightGBM": (

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

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,

                    verbosity=-1,
                )
            ),
        }


        # ====================================================
        # E = CLEARANCE
        #
        # ADD ROBUST LOSS VARIANTS
        # ====================================================

        if task == "e":


            # ------------------------------------------------
            # PSEUDO-HUBER XGBOOST
            # ------------------------------------------------

            models[
                "Pseudo-Huber XGBoost"
            ] = (

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
                        "reg:pseudohubererror"
                    ),

                    tree_method="hist",

                    device=(
                        xgboost_device
                    ),

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,
                )
            )


            # ------------------------------------------------
            # HUBER CATBOOST
            # ------------------------------------------------

            models[
                "Huber CatBoost"
            ] = (

                CatBoostRegressor(

                    iterations=600,

                    depth=7,

                    learning_rate=0.05,

                    loss_function=(
                        "Huber:delta=10.0"
                    ),

                    task_type=(
                        catboost_task_type
                    ),

                    random_seed=(
                        RANDOM_STATE
                    ),

                    verbose=False,

                    allow_writing_files=False,
                )
            )


            # ------------------------------------------------
            # HUBER LIGHTGBM
            # ------------------------------------------------

            models[
                "Huber LightGBM"
            ] = (

                LGBMRegressor(

                    n_estimators=600,

                    num_leaves=31,

                    max_depth=-1,

                    learning_rate=0.05,

                    subsample=0.85,

                    colsample_bytree=0.80,

                    reg_alpha=0.05,

                    reg_lambda=1.50,

                    objective="huber",

                    random_state=(
                        RANDOM_STATE
                    ),

                    n_jobs=-1,

                    verbosity=-1,
                )
            )


        return models


    raise ValueError(

        "task_type must be "
        "classification or regression."
    )


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_valid,
    task_type,
):


    if task_type == "classification":

        return (

            model
            .predict_proba(
                X_valid
            )[:, 1]
        )


    return model.predict(
        X_valid
    )


# ============================================================
# 5. CONVERT MODEL NAME TO WEIGHT KEY
# ============================================================

def get_weight_key(
    model_name,
):

    return (

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


# ============================================================
# 6. RUN EXPERIMENT
# ============================================================

def run_experiment(
    task,
    use_gpu=False,
):


    # ========================================================
    # CONFIG
    # ========================================================

    config = get_task_config(
        task
    )


    task = config[
        "task"
    ]


    print(
        "\n"
        + "=" * 70
    )


    print(

        f"{config['name']} "

        f"({config['dataset']}) "

        f"- {config['task_type']}"
    )


    print(
        "=" * 70
    )


    # ========================================================
    # OUTPUT FOLDERS
    # ========================================================

    (
        results_folder,
        saved_models_folder,

    ) = create_output_folders(
        task
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    data = load_data(
        task
    )


    X_train = data[
        "X_train"
    ]


    y_train = data[
        "y_train"
    ]


    X_valid = data[
        "X_valid"
    ]


    y_valid = data[
        "y_valid"
    ]


    # ========================================================
    # CLASSIFICATION TARGETS
    # ========================================================

    if (
        config[
            "task_type"
        ]
        == "classification"
    ):


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

        f"X_train shape: "

        f"{X_train.shape}"
    )


    print(

        f"X_valid shape: "

        f"{X_valid.shape}"
    )


    # ========================================================
    # TARGET
    #
    # NO LOG TRANSFORMATION
    # ========================================================

    y_train_for_model = (
        y_train
    )


    if (

        task == "e"

        and

        config[
            "task_type"
        ]
        == "regression"
    ):


        print(

            "\nE = Clearance"
        )


        print(

            "Using original target values."
        )


        print(

            "Testing standard + robust losses."
        )


        print(

            f"y_train range: "

            f"{np.min(y_train):.4f}"

            " to "

            f"{np.max(y_train):.4f}"
        )


    # ========================================================
    # IMPUTER
    # ========================================================

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


    # ========================================================
    # CREATE MODELS
    # ========================================================

    models = create_models(

        task=task,

        task_type=(
            config[
                "task_type"
            ]
        ),

        y_train=(
            y_train_for_model
        ),

        use_gpu=(
            use_gpu
        ),
    )


    print(

        f"\nNumber of models: "

        f"{len(models)}"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    validation_rows = []


    validation_predictions = {}


    # ========================================================
    # TRAIN EVERY MODEL
    # ========================================================

    for (

        model_name,
        model,

    ) in models.items():


        print(

            f"\nTraining "

            f"{model_name}..."
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(

            X_train_ready,

            y_train_for_model,
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predictions = (

            generate_predictions(

                model=model,

                X_valid=(
                    X_valid_ready
                ),

                task_type=(
                    config[
                        "task_type"
                    ]
                ),
            )
        )


        predictions = np.asarray(
            predictions
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = (

            calculate_metrics(

                y_true=(
                    y_valid
                ),

                predictions=(
                    predictions
                ),

                task_type=(
                    config[
                        "task_type"
                    ]
                ),
            )
        )


        # ----------------------------------------------------
        # SAVE METRICS
        # ----------------------------------------------------

        validation_rows.append(

            {

                "model": (
                    model_name
                ),

                **metrics,
            }
        )


        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        validation_predictions[
            model_name
        ] = predictions


        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_filename = (

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

            + ".joblib"
        )


        joblib.dump(

            model,

            saved_models_folder
            / model_filename,
        )


        # ----------------------------------------------------
        # PRINT METRICS
        # ----------------------------------------------------

        metric_text = ", ".join(

            f"{metric_name}="
            f"{metric_value:.5f}"

            for (

                metric_name,
                metric_value,

            ) in metrics.items()
        )


        print(

            f"Validation: "

            f"{metric_text}"
        )


    # ========================================================
    # DYNAMIC OPTIMIZED ENSEMBLE
    #
    # A/D/M/T/S -> 4 MODELS
    #
    # E -> 7 MODELS
    # ========================================================

    (

        weight_results,

        best_weight_row,

        best_ensemble_predictions,

    ) = search_ensemble_weights(


        y_valid=(
            y_valid
        ),


        predictions_by_model=(
            validation_predictions
        ),


        task_type=(
            config[
                "task_type"
            ]
        ),


        primary_metric=(
            config[
                "primary_metric"
            ]
        ),
    )


    # ========================================================
    # ENSEMBLE METRICS
    # ========================================================

    best_ensemble_metrics = (

        calculate_metrics(

            y_true=(
                y_valid
            ),

            predictions=(
                best_ensemble_predictions
            ),

            task_type=(
                config[
                    "task_type"
                ]
            ),
        )
    )


    # ========================================================
    # ENSEMBLE NAME
    # ========================================================

    ensemble_name = (

        f"{len(models)}-Model "

        f"Optimized Ensemble"
    )


    # ========================================================
    # ADD ENSEMBLE TO RESULTS
    # ========================================================

    validation_rows.append(

        {

            "model": (
                ensemble_name
            ),

            **best_ensemble_metrics,
        }
    )


    validation_predictions[
        ensemble_name
    ] = (

        best_ensemble_predictions
    )


    # ========================================================
    # VALIDATION METRICS DATAFRAME
    # ========================================================

    validation_metrics = (

        pd.DataFrame(
            validation_rows
        )
    )


    validation_metrics.to_csv(

        results_folder
        / "validation_metrics.csv",

        index=False,
    )


    # ========================================================
    # SAVE ENSEMBLE WEIGHTS
    # ========================================================

    weight_results.to_csv(

        results_folder
        / "ensemble_weight_results.csv",

        index=False,
    )


    # ========================================================
    # SAVE ALL VALIDATION PREDICTIONS
    # ========================================================

    prediction_data = {

        "y_true": (
            y_valid
        )
    }


    for (

        model_name,
        predictions,

    ) in validation_predictions.items():


        column_name = (

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

            + "_prediction"
        )


        prediction_data[
            column_name
        ] = predictions


    predictions_table = (

        pd.DataFrame(
            prediction_data
        )
    )


    predictions_table.to_csv(

        results_folder
        / "validation_predictions.csv",

        index=False,
    )


    # ========================================================
    # MODEL COMPARISON GRAPHS
    # ========================================================

    for metric in config[
        "metrics"
    ]:


        plot_model_comparison(

            validation_metrics=(
                validation_metrics
            ),

            metric=(
                metric
            ),

            endpoint_name=(
                config[
                    "name"
                ]
            ),

            output_path=(

                results_folder
                /

                f"{metric}"
                f"_model_comparison.png"
            ),
        )


    # ========================================================
    # FIND BEST SOLUTION
    # ========================================================

    best_solution_index = (

        get_best_index(

            validation_metrics[

                config[
                    "primary_metric"
                ]
            ],

            config[
                "primary_metric"
            ],
        )
    )


    best_solution_row = (

        validation_metrics

        .loc[
            best_solution_index
        ]

        .to_dict()
    )


    # ========================================================
    # EXTRACT DYNAMIC ENSEMBLE WEIGHTS
    # ========================================================

    ensemble_weights = {}


    for model_name in models.keys():


        weight_key = (

            get_weight_key(
                model_name
            )
        )


        if weight_key in best_weight_row:


            ensemble_weights[
                model_name
            ] = float(

                best_weight_row[
                    weight_key
                ]
            )


    # ========================================================
    # ENSEMBLE CONFIG
    # ========================================================

    ensemble_config = {


        "number_of_models": (

            len(models)
        ),


        "ensemble_name": (

            ensemble_name
        ),


        "weights": (

            ensemble_weights
        ),


        "selected_using": (

            config[
                "primary_metric"
            ]
        ),


        "validation_metrics": {

            metric: float(

                best_ensemble_metrics[
                    metric
                ]
            )

            for metric

            in config[
                "metrics"
            ]
        },
    }


    # ========================================================
    # SAVE ENSEMBLE CONFIG
    # ========================================================

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


    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    validation_summary = {


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

            config[
                "task_type"
            ]
        ),


        "primary_metric": (

            config[
                "primary_metric"
            ]
        ),


        "target_transformation": (

            "none"
        ),


        "base_models": (

            list(
                models.keys()
            )
        ),


        "best_validation_solution": (

            best_solution_row[
                "model"
            ]
        ),


        "best_validation_metrics": {

            metric: float(

                best_solution_row[
                    metric
                ]
            )

            for metric

            in config[
                "metrics"
            ]
        },


        "best_optimized_ensemble": (

            ensemble_config
        ),


        "gpu_requested": (

            use_gpu
        ),


        "test_set_used": (

            False
        ),
    }


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

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


    # ========================================================
    # PRINT VALIDATION RESULTS
    # ========================================================

    print(

        "\nValidation comparison"
    )


    print(

        validation_metrics

        .to_string(
            index=False
        )
    )


    # ========================================================
    # E INFORMATION
    # ========================================================

    if task == "e":


        print(

            "\nE = Clearance "
            "robust-loss experiment"
        )


        print(

            "Original target used "
            "(no log1p transformation)."
        )


    # ========================================================
    # PRINT OPTIMIZED WEIGHTS
    # ========================================================

    print(

        f"\nOptimized "

        f"{len(models)}-model "

        f"ensemble weights:"
    )


    for (

        model_name,
        weight,

    ) in ensemble_weights.items():


        print(

            f"  {model_name}: "

            f"{weight:.4f}"
        )


    # ========================================================
    # PRINT ENSEMBLE METRICS
    # ========================================================

    print(

        "\nOptimized ensemble metrics:"
    )


    for (

        metric_name,
        metric_value,

    ) in best_ensemble_metrics.items():


        print(

            f"  {metric_name}: "

            f"{metric_value:.5f}"
        )


    # ========================================================
    # PRINT BEST SOLUTION
    # ========================================================

    print(

        "\nBest validation solution: "

        f"{validation_summary['best_validation_solution']}"
    )


    print(

        f"\nResults saved in: "

        f"{results_folder}"
    )


    print(

        f"Models saved in: "

        f"{saved_models_folder}"
    )


    # ========================================================
    # RETURN
    # ========================================================

    return (
        validation_summary
    )