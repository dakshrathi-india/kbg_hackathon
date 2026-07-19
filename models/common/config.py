from pathlib import Path


# ============================================================
# GLOBAL SETTINGS
# ============================================================

RANDOM_STATE = 42

RESULTS_ROOT = Path("results")

SAVED_MODELS_ROOT = Path("saved_models")


# ============================================================
# ADMET TASK CONFIGURATION
# ============================================================

TASK_CONFIGS = {

    # --------------------------------------------------------
    # A = ABSORPTION
    # --------------------------------------------------------

    "a": {

        "name": "Absorption",

        "dataset": "Caco2_Wang",

        "task_type": "regression",

        "metrics": [
            "rmse",
            "r2",
        ],

        "primary_metric": "rmse",
    },


    # --------------------------------------------------------
    # D = DISTRIBUTION
    # --------------------------------------------------------

    "d": {

        "name": "Distribution",

        "dataset": "BBB_Martins",

        "task_type": "classification",

        "metrics": [
            "roc_auc",
            "auprc",
        ],

        "primary_metric": "roc_auc",
    },


    # --------------------------------------------------------
    # M = METABOLISM
    # --------------------------------------------------------

    "m": {

        "name": "Metabolism",

        "dataset": "CYP3A4_Veith",

        "task_type": "classification",

        "metrics": [
            "roc_auc",
            "auprc",
        ],

        "primary_metric": "auprc",
    },


    # --------------------------------------------------------
    # E = EXCRETION / CLEARANCE
    # --------------------------------------------------------

    "e": {

        "name": "Excretion",

        "dataset": "Clearance_Hepatocyte_AZ",

        "task_type": "regression",

        "metrics": [
            "rmse",
            "r2",
        ],

        "primary_metric": "rmse",
    },


    # --------------------------------------------------------
    # T = TOXICITY
    # --------------------------------------------------------

    "t": {

        "name": "Toxicity",

        "dataset": "AMES",

        "task_type": "classification",

        "metrics": [
            "roc_auc",
            "auprc",
        ],

        "primary_metric": "roc_auc",
    },


    # --------------------------------------------------------
    # S = AQUEOUS SOLUBILITY
    # --------------------------------------------------------

    "s": {

        "name": "Solubility",

        "dataset": "Solubility_AqSolDB",

        "task_type": "regression",

        "metrics": [
            "rmse",
            "r2",
        ],

        "primary_metric": "rmse",
    },
}


# ============================================================
# GET TASK CONFIG
# ============================================================

def get_task_config(
    task: str
) -> dict:

    """
    Validate a task code and return
    its configuration.

    Available tasks:

        a -> Absorption
        d -> Distribution
        m -> Metabolism
        e -> Excretion / Clearance
        t -> Toxicity
        s -> Solubility
    """

    task = (
        task
        .strip()
        .lower()
    )


    if task not in TASK_CONFIGS:

        raise ValueError(

            "Invalid task. "

            "Use one of: "

            "a, d, m, e, t, s."
        )


    config = (
        TASK_CONFIGS[
            task
        ].copy()
    )


    config[
        "task"
    ] = task


    return config