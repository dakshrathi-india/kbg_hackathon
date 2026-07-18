# 🧪 ADMET Prediction from SMILES

This project predicts five ADMET properties from molecular SMILES strings using traditional machine-learning models trained on Morgan fingerprints and RDKit descriptors.

## Current Scope

The project currently performs:

- TDC dataset extraction
- SMILES validation and canonicalization
- Scaffold-based train/validation splitting
- Morgan fingerprint generation
- RDKit descriptor generation
- Extra Trees training
- XGBoost training
- CatBoost training
- Weighted XGBoost–CatBoost ensembling
- Validation evaluation and graph generation

The test set is currently kept untouched so that a possible future GNN can be compared before final evaluation.

---

## Project Structure

```text
kbg_hack_ps1/
├── data_pipeline/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── data_loader.py
│   └── README.md
│
├── models/
│   ├── __init__.py
│   ├── README.md
│   └── common/
│       ├── __init__.py
│       ├── config.py
│       ├── metrics.py
│       ├── ensemble.py
│       ├── plotting.py
│       └── experiment.py
│
├── processed/
│   ├── a/
│   ├── d/
│   ├── m/
│   ├── e/
│   └── t/
│
├── results/
├── saved_models/
├── data/raw/
├── run_all.py
├── requirements.txt
└── README.md
```

---

## ADMET Endpoints

| Code | Endpoint | Dataset | Task |
|---|---|---|---|
| `a` | Absorption | Caco2_Wang | Regression |
| `d` | Distribution | BBB_Martins | Classification |
| `m` | Metabolism | CYP3A4_Veith | Classification |
| `e` | Excretion | Clearance_Hepatocyte_AZ | Regression |
| `t` | Toxicity | AMES | Classification |

---

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 1: Prepare the Data

Run from the main project folder:

```bash
python -m data_pipeline.prepare_data
```

This downloads and processes all five datasets.

The generated files are stored in:

```text
processed/a
processed/d
processed/m
processed/e
processed/t
```

Each endpoint contains:

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

## Step 2: Train and Validate Models

Run all endpoints on CPU:

```bash
python run_all.py
```

Run all endpoints using GPU for XGBoost and CatBoost:

```bash
python run_all.py --gpu
```

Run one endpoint:

```bash
python run_all.py --task a
python run_all.py --task d
python run_all.py --task m
python run_all.py --task e
python run_all.py --task t
```

---

## Model Workflow

For each endpoint:

```text
Load train and validation arrays
        ↓
Fit median imputer on training data
        ↓
Train Extra Trees
        ↓
Train XGBoost
        ↓
Train CatBoost
        ↓
Evaluate validation metrics
        ↓
Search XGBoost weights from 0.05 to 0.95
        ↓
Compare all models and the best ensemble
        ↓
Save outcomes
```

Classification endpoints report:

```text
ROC-AUC
AUPRC
```

Regression endpoints report:

```text
RMSE
R²
```

---

## Viewing Results

Each endpoint creates its own folder.

Example for toxicity:

```text
results/t/
├── validation_metrics.csv
├── ensemble_weight_results.csv
├── validation_predictions.csv
├── validation_summary.json
├── roc_auc_model_comparison.png
├── auprc_model_comparison.png
├── roc_auc_ensemble_weight_search.png
└── auprc_ensemble_weight_search.png
```

### Important result files

`validation_metrics.csv`

```text
Compares Extra Trees, XGBoost, CatBoost and the ensemble.
```

`ensemble_weight_results.csv`

```text
Shows the score produced by every tested weight combination.
```

`validation_summary.json`

```text
Shows the current best solution and its metrics.
```

`validation_predictions.csv`

```text
Stores model predictions for possible future GNN ensembling.
```

The PNG files show:

- Bar-chart model comparisons
- Line-graph ensemble-weight comparisons

---

## Saved Models

Trained models are stored in:

```text
saved_models/a
saved_models/d
saved_models/m
saved_models/e
saved_models/t
```

Each task folder contains:

```text
imputer.joblib
extra_trees.joblib
xgboost.joblib
catboost.joblib
ensemble_config.json
```

---

## Generated Folders

`results/` and `saved_models/` are automatically created when training starts.

To show these empty folders on GitHub before running the code, add:

```text
results/.gitkeep
saved_models/.gitkeep
```

Git does not track empty directories.

---

## Current Evaluation Rule

The validation set is currently used for:

- Comparing models
- Selecting ensemble weights
- Selecting the current best solution

The test set is not used yet.

Final test evaluation should happen only after the complete model architecture—including any possible GNN ensemble—has been fixed.