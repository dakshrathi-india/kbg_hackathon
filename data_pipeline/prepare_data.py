from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd

from tdc.benchmark_group import admet_group

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold


# ============================================================
# SETTINGS
# ============================================================

RAW_DATA_FOLDER = Path("data/raw")
PROCESSED_FOLDER = Path("processed")

SEED = 42

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# Validation fraction taken from TDC train_val.
VALID_FRACTION_OF_TRAIN_VAL = 0.125


# ============================================================
# TDC DATASETS
# ============================================================

DATASETS = {

    "a": "Caco2_Wang",

    "d": "BBB_Martins",

    "m": "CYP3A4_Veith",

    # E is now CLEARANCE again.
    "e": "Clearance_Hepatocyte_AZ",

    "t": "AMES",
}


DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "AromaticRingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "FormalCharge",
    "MolMR",
]


MORGAN_GENERATOR = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=True,
    )
)


# ============================================================
# 1. CANONICALIZE SMILES
# ============================================================

def canonicalize_smiles(smiles):

    if pd.isna(smiles):
        return None

    smiles = str(
        smiles
    ).strip()

    if not smiles:
        return None

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return None

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


# ============================================================
# 2. CLEAN DATAFRAME
# ============================================================

def clean_dataframe(
    dataframe,
    is_test=False,
):

    """
    Expected TDC columns:

        Drug_ID
        Drug
        Y
    """

    if "Drug" not in dataframe.columns:

        raise ValueError(
            "Drug/SMILES column not found. "
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )


    if "Y" not in dataframe.columns:

        raise ValueError(
            "Y/target column not found. "
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )


    cleaned = pd.DataFrame()


    # --------------------------------------------------------
    # DRUG ID
    # --------------------------------------------------------

    if "Drug_ID" in dataframe.columns:

        cleaned["drug_id"] = (

            dataframe[
                "Drug_ID"
            ]

            .astype(str)

            .to_numpy()
        )

    else:

        cleaned["drug_id"] = (

            np.arange(
                len(dataframe)
            )

            .astype(str)
        )


    # --------------------------------------------------------
    # SMILES
    # --------------------------------------------------------

    cleaned["smiles"] = (

        dataframe[
            "Drug"
        ]

        .astype(str)

        .to_numpy()
    )


    # --------------------------------------------------------
    # CANONICAL SMILES
    # --------------------------------------------------------

    cleaned[
        "canonical_smiles"
    ] = (

        cleaned[
            "smiles"
        ]

        .apply(
            canonicalize_smiles
        )
    )


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    cleaned["y"] = (

        pd.to_numeric(

            dataframe[
                "Y"
            ],

            errors="coerce",
        )

        .to_numpy()
    )


    # --------------------------------------------------------
    # INVALID ROWS
    # --------------------------------------------------------

    invalid_rows = (

        cleaned[
            "canonical_smiles"
        ].isna()

        |

        cleaned[
            "y"
        ].isna()
    )


    print(
        f"    Original rows: "
        f"{len(cleaned)}"
    )


    print(
        f"    Invalid/missing rows: "
        f"{invalid_rows.sum()}"
    )


    if (
        is_test
        and invalid_rows.any()
    ):

        raise ValueError(
            "Invalid SMILES or missing "
            "targets found in test set."
        )


    cleaned = (

        cleaned
        .loc[
            ~invalid_rows
        ]
        .copy()
    )


    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    if not is_test:

        rows_before = len(
            cleaned
        )


        cleaned = (

            cleaned
            .drop_duplicates(

                subset=[
                    "canonical_smiles",
                    "y",
                ],

                keep="first",
            )
        )


        print(
            "    Exact duplicates removed: "
            f"{rows_before - len(cleaned)}"
        )


    cleaned = (

        cleaned
        .reset_index(
            drop=True
        )
    )


    print(
        f"    Good rows remaining: "
        f"{len(cleaned)}"
    )


    return cleaned


# ============================================================
# 3. RANDOM SPLIT
# ============================================================

def random_split(
    dataframe,
    valid_fraction,
    seed=SEED,
):

    """
    Random train/validation split.

    This is intentionally used for E = Clearance
    to reproduce the earlier random-split setup.

    The random seed makes the split reproducible.
    """

    if not (
        0
        < valid_fraction
        < 1
    ):

        raise ValueError(
            "valid_fraction must be "
            "between 0 and 1."
        )


    rng = np.random.default_rng(
        seed
    )


    indices = np.arange(
        len(dataframe)
    )


    rng.shuffle(
        indices
    )


    valid_size = int(

        round(

            len(dataframe)
            * valid_fraction
        )
    )


    valid_indices = (

        indices[
            :valid_size
        ]
    )


    train_indices = (

        indices[
            valid_size:
        ]
    )


    train_dataframe = (

        dataframe
        .iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )


    valid_dataframe = (

        dataframe
        .iloc[
            valid_indices
        ]
        .reset_index(
            drop=True
        )
    )


    print(
        f"    Random train rows: "
        f"{len(train_dataframe)}"
    )


    print(
        f"    Random validation rows: "
        f"{len(valid_dataframe)}"
    )


    return (
        train_dataframe,
        valid_dataframe,
    )


# ============================================================
# 4. SCAFFOLD FUNCTIONS
# ============================================================

def get_scaffold(
    canonical_smiles
):

    molecule = Chem.MolFromSmiles(
        canonical_smiles
    )


    return (

        MurckoScaffold
        .MurckoScaffoldSmiles(

            mol=molecule,

            includeChirality=False,
        )
    )


def scaffold_split(
    dataframe,
    valid_fraction,
    seed=SEED,
):

    """
    Scaffold split used for endpoints
    other than E when needed.
    """

    scaffold_to_indices = defaultdict(
        list
    )


    for (
        row_index,
        smiles,
    ) in enumerate(

        dataframe[
            "canonical_smiles"
        ]
    ):

        scaffold = get_scaffold(
            smiles
        )


        scaffold_to_indices[
            scaffold
        ].append(
            row_index
        )


    scaffold_groups = list(

        scaffold_to_indices
        .values()
    )


    rng = np.random.default_rng(
        seed
    )


    rng.shuffle(
        scaffold_groups
    )


    scaffold_groups.sort(
        key=len
    )


    target_valid_size = int(

        round(

            len(dataframe)
            * valid_fraction
        )
    )


    train_indices = []

    valid_indices = []


    for scaffold_group in scaffold_groups:

        if (
            len(valid_indices)
            <
            target_valid_size
        ):

            valid_indices.extend(
                scaffold_group
            )

        else:

            train_indices.extend(
                scaffold_group
            )


    train_dataframe = (

        dataframe
        .iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )


    valid_dataframe = (

        dataframe
        .iloc[
            valid_indices
        ]
        .reset_index(
            drop=True
        )
    )


    return (
        train_dataframe,
        valid_dataframe,
    )


# ============================================================
# 5. CALCULATE DESCRIPTORS
# ============================================================

def calculate_descriptors(
    molecule
):

    return np.array(
        [

            Descriptors.MolWt(
                molecule
            ),

            Crippen.MolLogP(
                molecule
            ),

            rdMolDescriptors.CalcTPSA(
                molecule
            ),

            Lipinski.NumHDonors(
                molecule
            ),

            Lipinski.NumHAcceptors(
                molecule
            ),

            Lipinski.NumRotatableBonds(
                molecule
            ),

            Lipinski.RingCount(
                molecule
            ),

            Lipinski.NumAromaticRings(
                molecule
            ),

            Lipinski.HeavyAtomCount(
                molecule
            ),

            rdMolDescriptors.CalcFractionCSP3(
                molecule
            ),

            Chem.GetFormalCharge(
                molecule
            ),

            Crippen.MolMR(
                molecule
            ),
        ],

        dtype=np.float32,
    )


# ============================================================
# 6. CREATE FEATURES
# ============================================================

def create_features(
    canonical_smiles_list
):

    """
    2048 Morgan fingerprint
    +
    12 RDKit descriptors

    Total = 2060 features
    """

    number_of_molecules = len(
        canonical_smiles_list
    )


    total_features = (

        MORGAN_BITS

        +

        len(
            DESCRIPTOR_NAMES
        )
    )


    X = np.zeros(

        (

            number_of_molecules,

            total_features,
        ),

        dtype=np.float32,
    )


    for (
        row_index,
        smiles,
    ) in enumerate(

        canonical_smiles_list
    ):


        molecule = Chem.MolFromSmiles(
            smiles
        )


        if molecule is None:

            raise ValueError(
                "Invalid canonical "
                f"SMILES: {smiles}"
            )


        fingerprint = (

            MORGAN_GENERATOR
            .GetFingerprint(
                molecule
            )
        )


        fingerprint_array = (

            np.zeros(

                MORGAN_BITS,

                dtype=np.uint8,
            )
        )


        DataStructs.ConvertToNumpyArray(

            fingerprint,

            fingerprint_array,
        )


        descriptor_array = (

            calculate_descriptors(
                molecule
            )
        )


        X[
            row_index,
            :MORGAN_BITS
        ] = fingerprint_array


        X[
            row_index,
            MORGAN_BITS:
        ] = descriptor_array


        if (
            row_index + 1
        ) % 1000 == 0:

            print(

                "      Features generated "
                f"for {row_index + 1}/"
                f"{number_of_molecules}"
            )


    return X


# ============================================================
# 7. SAVE PROCESSED DATA
# ============================================================

def save_processed_data(

    folder_name,

    train_dataframe,

    valid_dataframe,

    test_dataframe,
):

    output_folder = (

        PROCESSED_FOLDER
        / folder_name
    )


    output_folder.mkdir(

        parents=True,

        exist_ok=True,
    )


    # --------------------------------------------------------
    # SAVE READABLE CSV
    # --------------------------------------------------------

    train_csv = (
        train_dataframe.copy()
    )

    valid_csv = (
        valid_dataframe.copy()
    )

    test_csv = (
        test_dataframe.copy()
    )


    train_csv[
        "split"
    ] = "train"


    valid_csv[
        "split"
    ] = "valid"


    test_csv[
        "split"
    ] = "test"


    good_data = pd.concat(

        [

            train_csv,

            valid_csv,

            test_csv,
        ],

        ignore_index=True,
    )


    good_data.to_csv(

        output_folder
        / "good_data.csv",

        index=False,
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    print(
        "\n    Generating training features..."
    )


    X_train = create_features(

        train_dataframe[
            "canonical_smiles"
        ].tolist()
    )


    print(
        "    Generating validation features..."
    )


    X_valid = create_features(

        valid_dataframe[
            "canonical_smiles"
        ].tolist()
    )


    print(
        "    Generating test features..."
    )


    X_test = create_features(

        test_dataframe[
            "canonical_smiles"
        ].tolist()
    )


    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    y_train = (

        train_dataframe[
            "y"
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    y_valid = (

        valid_dataframe[
            "y"
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    y_test = (

        test_dataframe[
            "y"
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    # --------------------------------------------------------
    # SAVE NPY FILES
    # --------------------------------------------------------

    np.save(

        output_folder
        / "X_train.npy",

        X_train,
    )


    np.save(

        output_folder
        / "y_train.npy",

        y_train,
    )


    np.save(

        output_folder
        / "X_valid.npy",

        X_valid,
    )


    np.save(

        output_folder
        / "y_valid.npy",

        y_valid,
    )


    np.save(

        output_folder
        / "X_test.npy",

        X_test,
    )


    np.save(

        output_folder
        / "y_test.npy",

        y_test,
    )


    print(
        "\n    DATA SAVED"
    )


    print(
        f"    Folder: "
        f"{output_folder}"
    )


    print(
        f"    X_train: "
        f"{X_train.shape}"
    )


    print(
        f"    X_valid: "
        f"{X_valid.shape}"
    )


    print(
        f"    X_test: "
        f"{X_test.shape}"
    )


# ============================================================
# 8. PREPARE E = CLEARANCE
# ============================================================

def prepare_clearance():

    """
    Prepare ONLY:

        E = Clearance_Hepatocyte_AZ

    Uses:

        TDC official train_val
        TDC official test

    The train_val portion is randomly split
    into train and validation.

    The official test set remains untouched.
    """

    print(
        "\n"
        + "=" * 70
    )


    print(
        "PREPARING E = EXCRETION / CLEARANCE"
    )


    print(
        "Dataset: Clearance_Hepatocyte_AZ"
    )


    print(
        "Split: RANDOM TRAIN / VALIDATION"
    )


    print(
        "=" * 70
    )


    RAW_DATA_FOLDER.mkdir(

        parents=True,

        exist_ok=True,
    )


    PROCESSED_FOLDER.mkdir(

        parents=True,

        exist_ok=True,
    )


    group = admet_group(

        path=str(
            RAW_DATA_FOLDER
        )
    )


    benchmark = group.get(

        "Clearance_Hepatocyte_AZ"
    )


    # --------------------------------------------------------
    # OFFICIAL TDC TRAIN_VAL
    # --------------------------------------------------------

    raw_train_val = (

        benchmark[
            "train_val"
        ]

        .copy()
    )


    # --------------------------------------------------------
    # OFFICIAL TDC TEST
    # --------------------------------------------------------

    raw_test = (

        benchmark[
            "test"
        ]

        .copy()
    )


    print(
        "\nCleaning Clearance train_val..."
    )


    clean_train_val = (

        clean_dataframe(

            raw_train_val,

            is_test=False,
        )
    )


    print(
        "\nCleaning official Clearance test..."
    )


    clean_test = (

        clean_dataframe(

            raw_test,

            is_test=True,
        )
    )


    # --------------------------------------------------------
    # RANDOM TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    print(
        "\nCreating RANDOM train/validation split..."
    )


    (

        train_dataframe,

        valid_dataframe,

    ) = random_split(

        clean_train_val,

        valid_fraction=(
            VALID_FRACTION_OF_TRAIN_VAL
        ),

        seed=SEED,
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print(
        "\nGenerating Clearance features..."
    )


    save_processed_data(

        folder_name="e",

        train_dataframe=(
            train_dataframe
        ),

        valid_dataframe=(
            valid_dataframe
        ),

        test_dataframe=(
            clean_test
        ),
    )


    print(
        "\n"
        + "=" * 70
    )


    print(
        "E = CLEARANCE HAS BEEN PREPARED SUCCESSFULLY"
    )


    print(
        "=" * 70
    )


# ============================================================
# 9. COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Prepare ADMET datasets."
        )
    )


    parser.add_argument(

        "--task",

        choices=[
            "e",
        ],

        required=True,

        help=(
            "Dataset to prepare. "
            "Currently use e for "
            "Clearance_Hepatocyte_AZ."
        ),
    )


    args = parser.parse_args()


    if args.task == "e":

        prepare_clearance()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()