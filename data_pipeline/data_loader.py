from pathlib import Path

import numpy as np


# Folder containing:
# processed/a
# processed/d
# processed/m
# processed/e
# processed/t
PROCESSED_FOLDER = Path("processed")


def load_data(task):
    """
    Load the prepared data for one ADMET task.

    Parameters
    ----------
    task : str
        One of:
        "a" -> Absorption
        "d" -> Distribution
        "m" -> Metabolism
        "e" -> Excretion
        "t" -> Toxicity

    Returns
    -------
    A dictionary containing:

        X_train
        y_train
        X_valid
        y_valid
        X_test
        y_test
    """

    # Convert inputs such as "A" or " a " into "a".
    task = task.strip().lower()

    valid_tasks = ["a", "d", "m", "e", "t"]

    if task not in valid_tasks:
        raise ValueError(
            "Invalid task. Please use one of: "
            "'a', 'd', 'm', 'e', or 't'."
        )

    task_folder = PROCESSED_FOLDER / task

    if not task_folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {task_folder}\n"
            "Run prepare_data.py first."
        )

    # Names of all files required for model training.
    file_paths = {
        "X_train": task_folder / "X_train.npy",
        "y_train": task_folder / "y_train.npy",

        "X_valid": task_folder / "X_valid.npy",
        "y_valid": task_folder / "y_valid.npy",

        "X_test": task_folder / "X_test.npy",
        "y_test": task_folder / "y_test.npy",
    }

    # Check that every required file exists.
    for file_name, file_path in file_paths.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}\n"
                "Run prepare_data.py again."
            )

    # Load NumPy arrays.
    data = {
        "X_train": np.load(file_paths["X_train"]),
        "y_train": np.load(file_paths["y_train"]),

        "X_valid": np.load(file_paths["X_valid"]),
        "y_valid": np.load(file_paths["y_valid"]),

        "X_test": np.load(file_paths["X_test"]),
        "y_test": np.load(file_paths["y_test"]),
    }

    # Verify that X and y have the same number of rows.
    if len(data["X_train"]) != len(data["y_train"]):
        raise ValueError(
            "X_train and y_train have different numbers of rows."
        )

    if len(data["X_valid"]) != len(data["y_valid"]):
        raise ValueError(
            "X_valid and y_valid have different numbers of rows."
        )

    if len(data["X_test"]) != len(data["y_test"]):
        raise ValueError(
            "X_test and y_test have different numbers of rows."
        )

    return data