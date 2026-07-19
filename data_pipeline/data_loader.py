from pathlib import Path

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

# Folder containing:
#
# processed/a
# processed/d
# processed/m
# processed/e
# processed/t
# processed/s

PROCESSED_FOLDER = Path("processed")


# ============================================================
# LOAD DATA
# ============================================================

def load_data(task):
    """
    Load the prepared data for one ADMET task.

    Parameters
    ----------
    task : str

        "a" -> Absorption
               Caco2_Wang

        "d" -> Distribution
               BBB_Martins

        "m" -> Metabolism
               CYP3A4_Veith

        "e" -> Excretion / Clearance
               Clearance_Hepatocyte_AZ

        "t" -> Toxicity
               AMES

        "s" -> Solubility
               Solubility_AqSolDB


    Returns
    -------
    Dictionary containing:

        X_train
        y_train

        X_valid
        y_valid

        X_test
        y_test
    """

    # ========================================================
    # NORMALIZE TASK NAME
    # ========================================================

    # Convert:
    #
    # "E"   -> "e"
    # " e " -> "e"

    task = (
        task
        .strip()
        .lower()
    )


    # ========================================================
    # VALID TASKS
    # ========================================================

    valid_tasks = [
        "a",
        "d",
        "m",
        "e",
        "t",
        "s",
    ]


    if task not in valid_tasks:

        raise ValueError(
            "Invalid task. "
            "Please use one of: "
            "'a', 'd', 'm', 'e', 't', or 's'."
        )


    # ========================================================
    # TASK FOLDER
    # ========================================================

    task_folder = (
        PROCESSED_FOLDER
        / task
    )


    if not task_folder.exists():

        raise FileNotFoundError(

            f"Folder not found: "
            f"{task_folder}\n"

            "Run prepare_data.py first."
        )


    # ========================================================
    # REQUIRED FILES
    # ========================================================

    file_paths = {

        "X_train": (
            task_folder
            / "X_train.npy"
        ),

        "y_train": (
            task_folder
            / "y_train.npy"
        ),


        "X_valid": (
            task_folder
            / "X_valid.npy"
        ),

        "y_valid": (
            task_folder
            / "y_valid.npy"
        ),


        "X_test": (
            task_folder
            / "X_test.npy"
        ),

        "y_test": (
            task_folder
            / "y_test.npy"
        ),
    }


    # ========================================================
    # CHECK FILES EXIST
    # ========================================================

    for (
        file_name,
        file_path,
    ) in file_paths.items():


        if not file_path.exists():

            raise FileNotFoundError(

                f"Missing file: "
                f"{file_path}\n"

                f"Required for: "
                f"{file_name}\n"

                "Run prepare_data.py "
                "for this endpoint first."
            )


    # ========================================================
    # LOAD NUMPY ARRAYS
    # ========================================================

    data = {

        "X_train": np.load(
            file_paths[
                "X_train"
            ]
        ),

        "y_train": np.load(
            file_paths[
                "y_train"
            ]
        ),


        "X_valid": np.load(
            file_paths[
                "X_valid"
            ]
        ),

        "y_valid": np.load(
            file_paths[
                "y_valid"
            ]
        ),


        "X_test": np.load(
            file_paths[
                "X_test"
            ]
        ),

        "y_test": np.load(
            file_paths[
                "y_test"
            ]
        ),
    }


    # ========================================================
    # VERIFY TRAIN DATA
    # ========================================================

    if (
        len(
            data[
                "X_train"
            ]
        )
        !=
        len(
            data[
                "y_train"
            ]
        )
    ):

        raise ValueError(
            "X_train and y_train "
            "have different numbers "
            "of rows."
        )


    # ========================================================
    # VERIFY VALIDATION DATA
    # ========================================================

    if (
        len(
            data[
                "X_valid"
            ]
        )
        !=
        len(
            data[
                "y_valid"
            ]
        )
    ):

        raise ValueError(
            "X_valid and y_valid "
            "have different numbers "
            "of rows."
        )


    # ========================================================
    # VERIFY TEST DATA
    # ========================================================

    if (
        len(
            data[
                "X_test"
            ]
        )
        !=
        len(
            data[
                "y_test"
            ]
        )
    ):

        raise ValueError(
            "X_test and y_test "
            "have different numbers "
            "of rows."
        )


    # ========================================================
    # PRINT LOADED DATASET INFO
    # ========================================================

    print(
        f"Loaded task: "
        f"{task.upper()}"
    )

    print(
        f"Train: "
        f"{data['X_train'].shape}"
    )

    print(
        f"Validation: "
        f"{data['X_valid'].shape}"
    )

    print(
        f"Test: "
        f"{data['X_test'].shape}"
    )


    # ========================================================
    # RETURN DATA
    # ========================================================

    return data