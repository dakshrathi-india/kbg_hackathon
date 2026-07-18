# 🧪 ADMET Data Pipeline

The `data_pipeline` package prepares and loads the molecular datasets used by our ADMET machine-learning models.

It ensures that every model receives:

- The same cleaned molecules
- The same train/validation/test split
- The same Morgan fingerprints
- The same RDKit descriptors

---

## 📁 Package Structure

```text
data_pipeline/
├── __init__.py
├── prepare_data.py
├── data_loader.py
└── README.md
```

### `prepare_data.py`

Downloads, cleans, splits and converts the molecular datasets into model-ready numerical files.

### `data_loader.py`

Loads the already-prepared NumPy arrays and returns them to the ML model.

### `__init__.py`

Marks `data_pipeline` as a Python package so that functions can be imported using:

```python
from data_pipeline.data_loader import load_data
```

---

# 📊 Selected ADMET Datasets

| Code | ADMET Category | TDC Dataset | Problem Type |
|---|---|---|---|
| `a` | Absorption | `Caco2_Wang` | Regression |
| `d` | Distribution | `BBB_Martins` | Classification |
| `m` | Metabolism | `CYP3A4_Veith` | Classification |
| `e` | Excretion | `Clearance_Hepatocyte_AZ` | Regression |
| `t` | Toxicity | `AMES` | Classification |

Each dataset predicts a different biological property, so it is processed and trained independently.

---

# 🔄 Complete Data Flow

```text
TDC Dataset
     ↓
Read train_val and test data
     ↓
Validate and canonicalize SMILES
     ↓
Remove invalid and duplicate rows
     ↓
Scaffold split train_val
     ↓
Train set + Validation set
     ↓
Generate Morgan fingerprints
     ↓
Calculate RDKit descriptors
     ↓
Save CSV and NumPy files
     ↓
Load data using data_loader.py
     ↓
Train ML models
```

---

# 1️⃣ Downloading the Dataset

`prepare_data.py` downloads the original datasets using PyTDC.

The original files are cached inside:

```text
data/raw/
```

This folder contains the untouched downloaded data.

The cleaned and model-ready files are saved separately inside:

```text
processed/
```

Keeping raw and processed data separate allows us to regenerate the processed data whenever the preprocessing logic changes.

---

# 2️⃣ SMILES Cleaning and Canonicalization

The original dataset contains:

| Column | Meaning |
|---|---|
| `Drug_ID` | Molecule identifier |
| `Drug` | SMILES string |
| `Y` | Target value |

A SMILES string is first converted into an RDKit molecule:

```python
molecule = Chem.MolFromSmiles(smiles)
```

Invalid SMILES are removed from the training data.

Valid SMILES are converted into canonical SMILES:

```python
canonical_smiles = Chem.MolToSmiles(
    molecule,
    canonical=True,
    isomericSmiles=True
)
```

Canonicalization provides a consistent representation of the same molecule and helps detect duplicates.

---

# 3️⃣ Data Cleaning

The pipeline removes rows containing:

- Invalid or missing SMILES
- Missing target values
- Non-numeric target values
- Exact duplicate molecule-target pairs

An exact duplicate means:

```text
Same canonical SMILES + Same target value
```

The official test set is not silently modified because the number and order of test predictions must remain fixed.

---

# 4️⃣ Scaffold Split

TDC provides:

```text
train_val
test
```

The `train_val` data is further divided into:

```text
train
validation
```

The split is performed using Bemis–Murcko molecular scaffolds.

```text
Molecules with the same structural core
                 ↓
        Remain in the same split
```

This prevents closely related molecular structures from appearing in both training and validation data.

The approximate final split is:

```text
Training   ≈ 70%
Validation ≈ 10%
Test       ≈ 20%
```

Exact percentages may vary slightly because complete scaffold groups must remain together.

---

# 5️⃣ Morgan Fingerprints

Every molecule is converted into a 2048-bit Morgan fingerprint.

```python
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
```

Example:

```text
[0, 1, 0, 0, 1, 0, ..., 1]
```

Morgan fingerprints represent the presence of different molecular substructures.

---

# 6️⃣ RDKit Descriptors

The pipeline also calculates 12 molecular descriptors:

- Molecular weight
- LogP
- TPSA
- Hydrogen-bond donors
- Hydrogen-bond acceptors
- Rotatable bonds
- Ring count
- Aromatic ring count
- Heavy atom count
- Fraction Csp3
- Formal charge
- Molar refractivity

These descriptors provide overall physicochemical information about a molecule.

```text
Morgan fingerprint → structural information
RDKit descriptors  → physicochemical information
```

The final feature vector contains:

```text
2048 Morgan features
+
12 RDKit descriptors
=
2060 features per molecule
```

---

# 📦 Generated Files

The processed data is stored as:

```text
processed/
├── a/
├── d/
├── m/
├── e/
└── t/
```

Each folder contains:

```text
good_data.csv

X_train.npy
y_train.npy

X_valid.npy
y_valid.npy

X_test.npy
y_test.npy
```

---

## `good_data.csv`

The readable CSV contains:

| Column | Meaning |
|---|---|
| `drug_id` | Molecule identifier |
| `smiles` | Original SMILES |
| `canonical_smiles` | Standardized SMILES |
| `y` | Target value |
| `split` | Train, valid or test |

Example:

```csv
drug_id,smiles,canonical_smiles,y,split
D001,OCC,CCO,1,train
D002,CC(=O)O,CC(=O)O,0,valid
```

The CSV is useful for manual inspection and debugging.

---

## NumPy Files

The `X` files contain the input features:

```text
X_train.npy
X_valid.npy
X_test.npy
```

Example:

```python
X_train.shape
# (5000, 2060)
```

This means:

```text
5000 molecules
2060 features per molecule
```

The `y` files contain the corresponding target values:

```text
y_train.npy
y_valid.npy
y_test.npy
```

The row alignment is preserved:

```text
X_train[0] corresponds to y_train[0]
X_train[1] corresponds to y_train[1]
```

---

# ▶️ Running the Pipeline

Run all commands from the main project folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare all five datasets:

```bash
python -m data_pipeline.prepare_data
```

The first run may take some time because the datasets must be downloaded and molecular features must be generated.

---

# 📥 Loading Data in a Model

Use the shared loader:

```python
from data_pipeline.data_loader import load_data
```

Example for toxicity:

```python
data = load_data("t")

X_train = data["X_train"]
y_train = data["y_train"]

X_valid = data["X_valid"]
y_valid = data["y_valid"]

X_test = data["X_test"]
y_test = data["y_test"]
```

The available task codes are:

```python
load_data("a")  # Absorption
load_data("d")  # Distribution
load_data("m")  # Metabolism
load_data("e")  # Excretion
load_data("t")  # Toxicity
```

The loader does not create a new split or regenerate features. It only loads the fixed data created by `prepare_data.py`.

---

# ✅ Why Use a Shared Data Pipeline?

Without a shared pipeline, different model files might:

- Use different data splits
- Remove different molecules
- Generate different features
- Use different random seeds

This would make model comparison unfair.

Our approach guarantees:

```text
Same data
+
Same features
+
Same split
=
Fair model comparison
```

---

# 🚫 Not Handled by This Package

This package does not perform:

- Model training
- Hyperparameter tuning
- Model evaluation
- Ensemble weight selection
- Model saving
- Graph plotting
- Final SMILES prediction

These operations are handled by the model-training and inference modules.

---

# ⚠️ Important Note

Changing any of the following requires regenerating the processed data and retraining the models:

- Scaffold split seed
- Validation fraction
- Morgan radius
- Fingerprint size
- Descriptor list
- Cleaning rules

The processed data and trained models must always use the same pipeline configuration.