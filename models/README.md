# 🤖 ADMET Model Pipeline

The `models` package contains the shared training, validation, ensembling, plotting and model-saving logic for all five ADMET endpoints.

## Structure

```text
models/
├── __init__.py
├── README.md
└── common/
    ├── __init__.py
    ├── config.py
    ├── metrics.py
    ├── ensemble.py
    ├── plotting.py
    └── experiment.py
```

The project does not use separate `run.py` files for A, D, M, E and T. A single root-level `run_all.py` calls the common experiment function with the required task code.

---

## Models Compared

For every endpoint, the pipeline trains and validates:

- Extra Trees
- XGBoost
- CatBoost
- Weighted XGBoost–CatBoost ensemble

The ensemble searches XGBoost weights from `0.05` to `0.95` in steps of `0.05`.

CatBoost receives the remaining weight:

```text
ensemble = xgb_weight × XGBoost prediction
         + (1 - xgb_weight) × CatBoost prediction
```

Weights are selected using validation data only.

---

## Endpoint Configuration

| Code | Endpoint | Dataset | Type | Metrics | Primary metric |
|---|---|---|---|---|---|
| `a` | Absorption | Caco2_Wang | Regression | RMSE, R² | RMSE |
| `d` | Distribution | BBB_Martins | Classification | ROC-AUC, AUPRC | ROC-AUC |
| `m` | Metabolism | CYP3A4_Veith | Classification | ROC-AUC, AUPRC | AUPRC |
| `e` | Excretion | Clearance_Hepatocyte_AZ | Regression | RMSE, R² | RMSE |
| `t` | Toxicity | AMES | Classification | ROC-AUC, AUPRC | ROC-AUC |

The primary metric selects the best ensemble weight and best validation solution. Both required metrics are still reported.

---

## File Responsibilities

### `config.py`

Stores:

- Endpoint information
- Classification or regression type
- Evaluation metrics
- Primary selection metric
- Random seed
- Output paths
- Ensemble weights

### `metrics.py`

Calculates:

```text
Classification → ROC-AUC and AUPRC
Regression     → RMSE and R²
```

### `ensemble.py`

Combines XGBoost and CatBoost validation predictions for every allowed weight and selects the best combination.

### `plotting.py`

Creates:

- Bar charts comparing all candidate solutions
- Line graphs showing performance across ensemble weights

### `experiment.py`

Runs one complete endpoint experiment:

```text
Load fixed train and validation data
        ↓
Fit median imputer on training data only
        ↓
Train Extra Trees, XGBoost and CatBoost
        ↓
Generate validation predictions
        ↓
Calculate validation metrics
        ↓
Search ensemble weights
        ↓
Save models, metrics, predictions and graphs
```

The test set is deliberately not evaluated at this stage.

---

## Generated Results

The code automatically creates:

```text
results/<task>/
├── validation_metrics.csv
├── ensemble_weight_results.csv
├── validation_predictions.csv
├── validation_summary.json
├── <metric>_model_comparison.png
└── <metric>_ensemble_weight_search.png
```

### `validation_metrics.csv`

Compares:

- Extra Trees
- XGBoost
- CatBoost
- Best weighted ensemble

### `ensemble_weight_results.csv`

Stores the validation metric for every tested XGBoost and CatBoost weight.

### `validation_predictions.csv`

Stores the actual validation predictions from every model. This can later be used for GNN ensembling.

### `validation_summary.json`

Stores:

- Endpoint name
- Dataset
- Primary metric
- Current best solution
- Best validation metrics
- Best XGBoost–CatBoost weights
- Confirmation that the test set was not used

---

## Saved Models

The code automatically creates:

```text
saved_models/<task>/
├── imputer.joblib
├── extra_trees.joblib
├── xgboost.joblib
├── catboost.joblib
└── ensemble_config.json
```

`ensemble_config.json` stores the best XGBoost and CatBoost validation weights.

---

## Running Experiments

Run every endpoint:

```bash
python run_all.py
```

Run one endpoint:

```bash
python run_all.py --task t
```

Use GPU for XGBoost and CatBoost:

```bash
python run_all.py --gpu
```

Extra Trees continues to run on CPU.

---

## Important Rules

- Every model uses the same split from `data_pipeline.data_loader`.
- Imputation is fitted only on training data.
- Classification metrics use probabilities, not hard labels.
- Ensemble weights are selected only using validation data.
- Test arrays remain untouched until the final architecture is complete.