from pathlib import Path
from collections import defaultdict

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

# TDC test contains approximately 20% of the complete data.
# 12.5% of the remaining train_val gives approximately:
# 70% train, 10% validation, 20% test.
VALID_FRACTION_OF_TRAIN_VAL = 0.125


# Folder name -> TDC dataset name
DATASETS = {
    "a": "Caco2_Wang",
    "d": "BBB_Martins",
    "m": "CYP3A4_Veith",
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


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
    includeChirality=True,
)


# ============================================================
# 1. CANONICALIZE SMILES
# ============================================================

def canonicalize_smiles(smiles):
    """
    Convert a SMILES string into canonical SMILES.

    Returns None when the SMILES is missing or invalid.
    """

    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    molecule = Chem.MolFromSmiles(smiles)

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

def clean_dataframe(dataframe, is_test=False):
    """
    Expected TDC columns:

    Drug_ID -> molecule ID
    Drug    -> SMILES
    Y       -> target value
    """

    if "Drug" not in dataframe.columns:
        raise ValueError(
            f"Drug/SMILES column not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    if "Y" not in dataframe.columns:
        raise ValueError(
            f"Y/target column not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    cleaned = pd.DataFrame()

    if "Drug_ID" in dataframe.columns:
        cleaned["drug_id"] = (
            dataframe["Drug_ID"]
            .astype(str)
            .to_numpy()
        )
    else:
        cleaned["drug_id"] = (
            np.arange(len(dataframe))
            .astype(str)
        )

    cleaned["smiles"] = (
        dataframe["Drug"]
        .astype(str)
        .to_numpy()
    )

    cleaned["canonical_smiles"] = (
        cleaned["smiles"]
        .apply(canonicalize_smiles)
    )

    cleaned["y"] = pd.to_numeric(
        dataframe["Y"],
        errors="coerce",
    ).to_numpy()

    invalid_rows = (
        cleaned["canonical_smiles"].isna()
        | cleaned["y"].isna()
    )

    print(f"    Original rows: {len(cleaned)}")
    print(f"    Invalid/missing rows: {invalid_rows.sum()}")

    if is_test and invalid_rows.any():
        raise ValueError(
            "Invalid SMILES or missing targets were found in "
            "the official test set. Test rows were not removed "
            "because the test order must remain unchanged."
        )

    cleaned = cleaned.loc[~invalid_rows].copy()

    # Remove exact duplicate molecule-target rows from train_val.
    # Do not change the official test set.
    if not is_test:
        rows_before = len(cleaned)

        cleaned = cleaned.drop_duplicates(
            subset=["canonical_smiles", "y"],
            keep="first",
        )

        print(
            f"    Exact duplicates removed: "
            f"{rows_before - len(cleaned)}"
        )

    cleaned = cleaned.reset_index(drop=True)

    print(f"    Good rows remaining: {len(cleaned)}")

    return cleaned


# ============================================================
# 3. GENERATE BEMIS-MURCKO SCAFFOLD
# ============================================================

def get_scaffold(canonical_smiles):
    """
    Generate the core molecular scaffold for one molecule.
    """

    molecule = Chem.MolFromSmiles(canonical_smiles)

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )


# ============================================================
# 4. SCAFFOLD SPLIT: TRAIN_VAL -> TRAIN + VALID
# ============================================================

def scaffold_split(
    dataframe,
    valid_fraction=VALID_FRACTION_OF_TRAIN_VAL,
    seed=SEED,
):
    """
    Split train_val into train and validation.

    Molecules having the same scaffold are always placed
    in the same split.
    """

    scaffold_to_indices = defaultdict(list)

    for row_index, smiles in enumerate(
        dataframe["canonical_smiles"]
    ):
        scaffold = get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(row_index)

    scaffold_groups = list(
        scaffold_to_indices.values()
    )

    # Shuffle groups reproducibly.
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_groups)

    # Smaller scaffold groups are placed into validation first.
    scaffold_groups.sort(key=len)

    target_valid_size = int(
        round(len(dataframe) * valid_fraction)
    )

    train_indices = []
    valid_indices = []

    for scaffold_group in scaffold_groups:

        if len(valid_indices) < target_valid_size:
            valid_indices.extend(scaffold_group)
        else:
            train_indices.extend(scaffold_group)

    train_dataframe = (
        dataframe
        .iloc[train_indices]
        .reset_index(drop=True)
    )

    valid_dataframe = (
        dataframe
        .iloc[valid_indices]
        .reset_index(drop=True)
    )

    # Check that the same scaffold does not occur in both sets.
    train_scaffolds = set(
        train_dataframe["canonical_smiles"]
        .apply(get_scaffold)
    )

    valid_scaffolds = set(
        valid_dataframe["canonical_smiles"]
        .apply(get_scaffold)
    )

    common_scaffolds = (
        train_scaffolds.intersection(valid_scaffolds)
    )

    if common_scaffolds:
        raise RuntimeError(
            "Scaffold leakage detected between "
            "training and validation data."
        )

    print(f"    Train rows: {len(train_dataframe)}")
    print(f"    Valid rows: {len(valid_dataframe)}")
    print(f"    Common scaffolds: {len(common_scaffolds)}")

    return train_dataframe, valid_dataframe


# ============================================================
# 5. CALCULATE RDKIT DESCRIPTORS
# ============================================================

def calculate_descriptors(molecule):
    """
    Calculate 12 RDKit molecular descriptors.
    """

    return np.array(
        [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            Lipinski.NumAromaticRings(molecule),
            Lipinski.HeavyAtomCount(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            Chem.GetFormalCharge(molecule),
            Crippen.MolMR(molecule),
        ],
        dtype=np.float32,
    )


# ============================================================
# 6. GENERATE MORGAN + DESCRIPTOR FEATURES
# ============================================================

def create_features(canonical_smiles_list):
    """
    Final feature order:

    First 2048 columns -> Morgan fingerprint
    Last 12 columns    -> RDKit descriptors

    Total features = 2060
    """

    number_of_molecules = len(
        canonical_smiles_list
    )

    total_features = (
        MORGAN_BITS + len(DESCRIPTOR_NAMES)
    )

    X = np.zeros(
        (number_of_molecules, total_features),
        dtype=np.float32,
    )

    for row_index, smiles in enumerate(
        canonical_smiles_list
    ):
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            raise ValueError(
                f"Invalid canonical SMILES: {smiles}"
            )

        # Morgan fingerprint
        fingerprint = (
            MORGAN_GENERATOR
            .GetFingerprint(molecule)
        )

        fingerprint_array = np.zeros(
            MORGAN_BITS,
            dtype=np.uint8,
        )

        DataStructs.ConvertToNumpyArray(
            fingerprint,
            fingerprint_array,
        )

        # RDKit descriptors
        descriptor_array = calculate_descriptors(
            molecule
        )

        X[row_index, :MORGAN_BITS] = (
            fingerprint_array
        )

        X[row_index, MORGAN_BITS:] = (
            descriptor_array
        )

        if (row_index + 1) % 1000 == 0:
            print(
                f"      Features generated for "
                f"{row_index + 1}/"
                f"{number_of_molecules}"
            )

    return X


# ============================================================
# 7. SAVE CSV AND NPY FILES
# ============================================================

def save_processed_data(
    folder_name,
    train_dataframe,
    valid_dataframe,
    test_dataframe,
):
    """
    Save files inside:

    processed/a
    processed/d
    processed/m
    processed/e
    processed/t
    """

    output_folder = (
        PROCESSED_FOLDER / folder_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create one readable CSV containing all good data
    # --------------------------------------------------------

    train_csv = train_dataframe.copy()
    valid_csv = valid_dataframe.copy()
    test_csv = test_dataframe.copy()

    train_csv["split"] = "train"
    valid_csv["split"] = "valid"
    test_csv["split"] = "test"

    good_data = pd.concat(
        [
            train_csv,
            valid_csv,
            test_csv,
        ],
        ignore_index=True,
    )

    good_data.to_csv(
        output_folder / "good_data.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Generate numerical features
    # --------------------------------------------------------

    print("    Generating training features...")

    X_train = create_features(
        train_dataframe[
            "canonical_smiles"
        ].tolist()
    )

    print("    Generating validation features...")

    X_valid = create_features(
        valid_dataframe[
            "canonical_smiles"
        ].tolist()
    )

    print("    Generating test features...")

    X_test = create_features(
        test_dataframe[
            "canonical_smiles"
        ].tolist()
    )

    y_train = train_dataframe["y"].to_numpy(
        dtype=np.float32
    )

    y_valid = valid_dataframe["y"].to_numpy(
        dtype=np.float32
    )

    y_test = test_dataframe["y"].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Save NPY files
    # --------------------------------------------------------

    np.save(
        output_folder / "X_train.npy",
        X_train,
    )

    np.save(
        output_folder / "y_train.npy",
        y_train,
    )

    np.save(
        output_folder / "X_valid.npy",
        X_valid,
    )

    np.save(
        output_folder / "y_valid.npy",
        y_valid,
    )

    np.save(
        output_folder / "X_test.npy",
        X_test,
    )

    np.save(
        output_folder / "y_test.npy",
        y_test,
    )

    print(f"    Saved inside: {output_folder}")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_valid: {X_valid.shape}")
    print(f"    X_test:  {X_test.shape}")


# ============================================================
# 8. MAIN PREPARE_DATA FUNCTION
# ============================================================

def prepare_data():
    """
    Complete data-preparation pipeline for all five datasets.
    """

    RAW_DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # TDC downloads/caches the original data here.
    group = admet_group(
        path=str(RAW_DATA_FOLDER)
    )

    for folder_name, dataset_name in DATASETS.items():

        print("\n" + "=" * 60)
        print(
            f"Preparing {dataset_name} "
            f"inside processed/{folder_name}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Receive data from TDC
        # ----------------------------------------------------

        benchmark = group.get(dataset_name)

        raw_train_val = (
            benchmark["train_val"]
            .copy()
        )

        raw_test = (
            benchmark["test"]
            .copy()
        )

        # ----------------------------------------------------
        # Canonicalize and clean
        # ----------------------------------------------------

        print("\n  Cleaning train_val data...")

        clean_train_val = clean_dataframe(
            raw_train_val,
            is_test=False,
        )

        print("\n  Cleaning test data...")

        clean_test = clean_dataframe(
            raw_test,
            is_test=True,
        )

        # ----------------------------------------------------
        # Scaffold split train_val into train and valid
        # ----------------------------------------------------

        print("\n  Creating scaffold split...")

        train_dataframe, valid_dataframe = (
            scaffold_split(
                clean_train_val,
                valid_fraction=(
                    VALID_FRACTION_OF_TRAIN_VAL
                ),
                seed=SEED,
            )
        )

        # ----------------------------------------------------
        # Generate features and save files
        # ----------------------------------------------------

        print("\n  Saving processed data...")

        save_processed_data(
            folder_name=folder_name,
            train_dataframe=train_dataframe,
            valid_dataframe=valid_dataframe,
            test_dataframe=clean_test,
        )

    print("\n" + "=" * 60)
    print("ALL FIVE DATASETS HAVE BEEN PREPARED")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()