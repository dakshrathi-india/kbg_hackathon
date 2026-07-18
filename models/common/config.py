from pathlib import Path


RANDOM_STATE = 42

RESULTS_ROOT = Path("results")
SAVED_MODELS_ROOT = Path("saved_models")


# XGBoost receives each weight below.
# CatBoost receives: 1 - xgboost_weight
ENSEMBLE_WEIGHTS = [
    round(step * 0.05, 2)
    for step in range(1, 20)
]


TASK_CONFIGS = {
    "a": {
        "name": "Absorption",
        "dataset": "Caco2_Wang",
        "task_type": "regression",
        "metrics": ["rmse", "r2"],
        "primary_metric": "rmse",
    },

    "d": {
        "name": "Distribution",
        "dataset": "BBB_Martins",
        "task_type": "classification",
        "metrics": ["roc_auc", "auprc"],
        "primary_metric": "roc_auc",
    },

    "m": {
        "name": "Metabolism",
        "dataset": "CYP3A4_Veith",
        "task_type": "classification",
        "metrics": ["roc_auc", "auprc"],
        "primary_metric": "auprc",
    },

    "e": {
        "name": "Excretion",
        "dataset": "Clearance_Hepatocyte_AZ",
        "task_type": "regression",
        "metrics": ["rmse", "r2"],
        "primary_metric": "rmse",
    },

    "t": {
        "name": "Toxicity",
        "dataset": "AMES",
        "task_type": "classification",
        "metrics": ["roc_auc", "auprc"],
        "primary_metric": "roc_auc",
    },
}


def get_task_config(task: str) -> dict:
    """
    Validate a task code and return its configuration.
    """

    task = task.strip().lower()

    if task not in TASK_CONFIGS:
        raise ValueError(
            "Invalid task. Use one of: a, d, m, e, t."
        )

    config = TASK_CONFIGS[task].copy()
    config["task"] = task

    return config