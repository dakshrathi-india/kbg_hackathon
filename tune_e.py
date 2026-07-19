import json
from pathlib import Path

import joblib
import numpy as np
import optuna

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_pipeline.data_loader import load_data
from models.common.config import RANDOM_STATE


# ============================================================
# SETTINGS
# ============================================================

N_TRIALS = 50

OUTPUT_FOLDER = Path(
    "results/e/hyperparameter_tuning"
)

MODEL_FOLDER = Path(
    "saved_models/e/tuned"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD E = CLEARANCE
# ============================================================

print("=" * 70)
print("E = CLEARANCE HYPERPARAMETER TUNING")
print("=" * 70)

data = load_data("e")

X_train = data["X_train"]
y_train = data["y_train"]

X_valid = data["X_valid"]
y_valid = data["y_valid"]


print(
    "X_train:",
    X_train.shape
)

print(
    "X_valid:",
    X_valid.shape
)


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True,
)

X_train_ready = imputer.fit_transform(
    X_train
)

X_valid_ready = imputer.transform(
    X_valid
)

joblib.dump(
    imputer,
    MODEL_FOLDER / "imputer.joblib",
)


# ============================================================
# METRICS
# ============================================================

def calculate_regression_metrics(
    y_true,
    predictions,
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ============================================================
# XGBOOST OBJECTIVE
# ============================================================

def xgboost_objective(
    trial,
):

    params = {

        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                300,
                1500,
                step=100,
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                3,
                10,
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),

        "min_child_weight":
            trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),

        "subsample":
            trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

        "colsample_bytree":
            trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),

        "reg_alpha":
            trial.suggest_float(
                "reg_alpha",
                1e-4,
                5.0,
                log=True,
            ),

        "reg_lambda":
            trial.suggest_float(
                "reg_lambda",
                1e-3,
                10.0,
                log=True,
            ),

        "gamma":
            trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),
    }


    model = XGBRegressor(

        **params,

        objective="reg:squarederror",

        eval_metric="rmse",

        tree_method="hist",

        device="cuda",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )


    model.fit(

        X_train_ready,

        y_train,
    )


    predictions = model.predict(

        X_valid_ready
    )


    rmse = np.sqrt(

        mean_squared_error(

            y_valid,

            predictions,
        )
    )


    return float(
        rmse
    )


# ============================================================
# CATBOOST OBJECTIVE
# ============================================================

def catboost_objective(
    trial,
):

    params = {

        "iterations":
            trial.suggest_int(
                "iterations",
                300,
                1500,
                step=100,
            ),

        "depth":
            trial.suggest_int(
                "depth",
                4,
                10,
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),

        "l2_leaf_reg":
            trial.suggest_float(
                "l2_leaf_reg",
                1.0,
                20.0,
                log=True,
            ),

        "random_strength":
            trial.suggest_float(
                "random_strength",
                0.0,
                5.0,
            ),

        "bagging_temperature":
            trial.suggest_float(
                "bagging_temperature",
                0.0,
                5.0,
            ),
    }


    model = CatBoostRegressor(

        **params,

        loss_function="RMSE",

        task_type="GPU",

        random_seed=RANDOM_STATE,

        verbose=False,

        allow_writing_files=False,
    )


    model.fit(

        X_train_ready,

        y_train,
    )


    predictions = model.predict(

        X_valid_ready
    )


    rmse = np.sqrt(

        mean_squared_error(

            y_valid,

            predictions,
        )
    )


    return float(
        rmse
    )


# ============================================================
# LIGHTGBM OBJECTIVE
# ============================================================

def lightgbm_objective(
    trial,
):

    params = {

        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                300,
                1500,
                step=100,
            ),

        "num_leaves":
            trial.suggest_int(
                "num_leaves",
                15,
                100,
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                3,
                12,
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),

        "min_child_samples":
            trial.suggest_int(
                "min_child_samples",
                5,
                50,
            ),

        "subsample":
            trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

        "colsample_bytree":
            trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),

        "reg_alpha":
            trial.suggest_float(
                "reg_alpha",
                1e-4,
                5.0,
                log=True,
            ),

        "reg_lambda":
            trial.suggest_float(
                "reg_lambda",
                1e-3,
                10.0,
                log=True,
            ),
    }


    model = LGBMRegressor(

        **params,

        objective="regression",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1,
    )


    model.fit(

        X_train_ready,

        y_train,
    )


    predictions = model.predict(

        X_valid_ready
    )


    rmse = np.sqrt(

        mean_squared_error(

            y_valid,

            predictions,
        )
    )


    return float(
        rmse
    )


# ============================================================
# RUN OPTUNA
# ============================================================

def run_study(
    name,
    objective,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"TUNING {name}"
    )

    print(
        "=" * 70
    )


    study = optuna.create_study(

        direction="minimize",

        study_name=name,

        sampler=(
            optuna.samplers.TPESampler(

                seed=RANDOM_STATE
            )
        ),
    )


    study.optimize(

        objective,

        n_trials=N_TRIALS,
    )


    print(

        f"\n{name} BEST RMSE: "

        f"{study.best_value:.5f}"
    )


    print(

        f"{name} BEST PARAMETERS:"
    )


    for (
        parameter,
        value,

    ) in study.best_params.items():

        print(

            f"  {parameter}: "

            f"{value}"
        )


    return study


# ============================================================
# TUNE XGBOOST
# ============================================================

xgb_study = run_study(

    "Tuned XGBoost",

    xgboost_objective,
)


# ============================================================
# TUNE CATBOOST
# ============================================================

cat_study = run_study(

    "Tuned CatBoost",

    catboost_objective,
)


# ============================================================
# TUNE LIGHTGBM
# ============================================================

lgbm_study = run_study(

    "Tuned LightGBM",

    lightgbm_objective,
)


# ============================================================
# TRAIN FINAL TUNED XGBOOST
# ============================================================

print(
    "\nTraining final tuned XGBoost..."
)


tuned_xgboost = XGBRegressor(

    **xgb_study.best_params,

    objective="reg:squarederror",

    eval_metric="rmse",

    tree_method="hist",

    device="cuda",

    random_state=RANDOM_STATE,

    n_jobs=-1,
)


tuned_xgboost.fit(

    X_train_ready,

    y_train,
)


xgb_predictions = (

    tuned_xgboost.predict(

        X_valid_ready
    )
)


xgb_metrics = (

    calculate_regression_metrics(

        y_valid,

        xgb_predictions,
    )
)


# ============================================================
# TRAIN FINAL TUNED CATBOOST
# ============================================================

print(
    "Training final tuned CatBoost..."
)


tuned_catboost = CatBoostRegressor(

    **cat_study.best_params,

    loss_function="RMSE",

    task_type="GPU",

    random_seed=RANDOM_STATE,

    verbose=False,

    allow_writing_files=False,
)


tuned_catboost.fit(

    X_train_ready,

    y_train,
)


cat_predictions = (

    tuned_catboost.predict(

        X_valid_ready
    )
)


cat_metrics = (

    calculate_regression_metrics(

        y_valid,

        cat_predictions,
    )
)


# ============================================================
# TRAIN FINAL TUNED LIGHTGBM
# ============================================================

print(
    "Training final tuned LightGBM..."
)


tuned_lightgbm = LGBMRegressor(

    **lgbm_study.best_params,

    objective="regression",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1,
)


tuned_lightgbm.fit(

    X_train_ready,

    y_train,
)


lgbm_predictions = (

    tuned_lightgbm.predict(

        X_valid_ready
    )
)


lgbm_metrics = (

    calculate_regression_metrics(

        y_valid,

        lgbm_predictions,
    )
)


# ============================================================
# SAVE FINAL MODELS
# ============================================================

joblib.dump(

    tuned_xgboost,

    MODEL_FOLDER
    / "tuned_xgboost.joblib",
)


joblib.dump(

    tuned_catboost,

    MODEL_FOLDER
    / "tuned_catboost.joblib",
)


joblib.dump(

    tuned_lightgbm,

    MODEL_FOLDER
    / "tuned_lightgbm.joblib",
)


# ============================================================
# SAVE BEST PARAMETERS
# ============================================================

best_parameters = {

    "XGBoost":
        xgb_study.best_params,

    "CatBoost":
        cat_study.best_params,

    "LightGBM":
        lgbm_study.best_params,
}


with open(

    OUTPUT_FOLDER
    / "best_parameters.json",

    "w",

    encoding="utf-8",

) as file:

    json.dump(

        best_parameters,

        file,

        indent=4,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "XGBoost": (
        xgb_metrics
    ),

    "CatBoost": (
        cat_metrics
    ),

    "LightGBM": (
        lgbm_metrics
    ),
}


with open(

    OUTPUT_FOLDER
    / "tuned_results.json",

    "w",

    encoding="utf-8",

) as file:

    json.dump(

        results,

        file,

        indent=4,
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL TUNED RESULTS"
)

print(
    "=" * 70
)


print(

    "\nTuned XGBoost:"

    f"\nRMSE = "
    f"{xgb_metrics['rmse']:.5f}"

    f"\nR²   = "
    f"{xgb_metrics['r2']:.5f}"
)


print(

    "\nTuned CatBoost:"

    f"\nRMSE = "
    f"{cat_metrics['rmse']:.5f}"

    f"\nR²   = "
    f"{cat_metrics['r2']:.5f}"
)


print(

    "\nTuned LightGBM:"

    f"\nRMSE = "
    f"{lgbm_metrics['rmse']:.5f}"

    f"\nR²   = "
    f"{lgbm_metrics['r2']:.5f}"
)


print(

    "\nCurrent ensemble baseline:"

    "\nRMSE = 42.88060"

    "\nR²   = 0.26675"
)