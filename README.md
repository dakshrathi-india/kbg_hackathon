# DELTA001 — SMILES to ADMET Prediction

A machine-learning web application that predicts important **ADMET properties** and **aqueous solubility** directly from a molecular **SMILES** string.

Developed by **Team DELTA001** for the **KBG Computational Biology Hackathon, IIT Mandi**.

---

## Live Project

- **Deployed Application:** https://delta001-admet.pages.dev
- **GitHub Repository:** https://github.com/dakshrathi-india/kbg_hackathon

> **Deployment note:** The React frontend is permanently hosted on Cloudflare Pages. During the hackathon, the FastAPI backend was exposed through a Cloudflare Quick Tunnel connected to a locally running backend. The website will open even when the backend is unavailable, but real predictions require the configured backend service to be running.

---

## Overview

A molecule may show promising biological activity and still fail as a drug because of poor:

- absorption;
- distribution;
- metabolism;
- excretion;
- toxicity; or
- aqueous solubility.

DELTA001 provides a preliminary molecular-screening system that accepts a SMILES string and predicts six pharmacokinetic, safety, and physicochemical properties.

The platform:

1. validates the submitted SMILES;
2. converts it into canonical SMILES;
3. generates Morgan fingerprints and RDKit molecular descriptors;
4. applies the saved preprocessing objects;
5. runs multiple trained machine-learning models;
6. averages predictions across three random seeds;
7. combines model-family outputs using endpoint-specific ensemble weights;
8. displays molecular predictions, descriptors, and structure information; and
9. stores recent predictions in a SQLite database.

---

## Predicted Endpoints

| Code | Category | Predicted Property | Dataset | Task |
|---|---|---|---|---|
| A | Absorption | Caco-2 Permeability | Caco2_Wang | Regression |
| D | Distribution | Blood-Brain Barrier Penetration | BBB_Martins | Classification |
| M | Metabolism | CYP3A4 Inhibition | CYP3A4_Veith | Classification |
| E | Excretion | Hepatocyte Clearance | Clearance_Hepatocyte_AZ | Regression |
| T | Toxicity | AMES Mutagenicity | AMES | Classification |
| S | Solubility | Aqueous Solubility | Solubility_AqSolDB | Regression |

---

## End-to-End Workflow

The project consists of two separate workflows:

1. **Model-development workflow**
2. **Web-inference workflow**

### Model-development workflow

```text
Endpoint dataset
      |
      v
SMILES and label validation
      |
      v
Canonicalization and duplicate handling
      |
      v
TDC held-out test split
      |
      v
Scaffold-based train-validation split
      |
      v
Morgan fingerprints + RDKit descriptors
      |
      v
Missing-value imputation
      |
      v
Train four model families using three seeds
      |
      v
Evaluate on validation set
      |
      v
Select model or ensemble weights
      |
      v
Evaluate final system once on held-out test set
      |
      v
Save models, imputers and ensemble configuration
```

### Web-inference workflow

```text
User enters SMILES
      |
      v
React frontend sends POST request
      |
      v
FastAPI validates and canonicalizes SMILES
      |
      v
Generate the same 2,060 features used in training
      |
      v
Apply endpoint-specific saved imputer
      |
      v
Load four model families × three seeds
      |
      v
Average predictions across seeds
      |
      v
Apply endpoint-specific ensemble weights
      |
      v
Return six predictions and molecular descriptors
      |
      v
Store prediction in SQLite history
```

---

## Dataset Splitting Strategy

The project uses a leakage-aware evaluation strategy.

### Held-out test set

For each endpoint, the test set supplied through the dataset workflow is kept completely separate from model development.

The held-out test portion is approximately:

```text
20% of the complete dataset
```

The test set is not used for:

- fitting the models;
- selecting hyperparameters;
- selecting ensemble weights; or
- choosing the final model family.

### Train-validation split

The remaining train-validation data is divided using a molecular scaffold split.

A validation fraction of:

```text
0.125 of the remaining train-validation data
```

produces an approximate overall division of:

```text
70% training
10% validation
20% testing
```

### Why scaffold splitting?

Random splitting can place structurally similar molecules in both the training and evaluation sets. This can result in overly optimistic performance.

Scaffold splitting separates molecules according to their core chemical structures and provides a more realistic evaluation on unseen molecular scaffolds.

---

## Data Cleaning

The data pipeline performs the following operations independently for every endpoint:

1. loads the endpoint dataset;
2. checks required columns;
3. removes invalid SMILES strings;
4. converts valid molecules into canonical SMILES;
5. removes exact duplicate molecules;
6. checks for missing labels;
7. detects conflicting labels among duplicate molecules;
8. records removed samples;
9. applies the required split strategy;
10. generates molecular features; and
11. saves processed arrays and metadata.

Canonicalization ensures that equivalent molecular representations are converted into a consistent SMILES format.

---

## Molecular Representation

Each molecule is represented using:

- **2,048-bit Morgan fingerprint**
- **12 RDKit molecular descriptors**
- **2,060 total input features**

### Morgan fingerprint configuration

The project uses a circular Morgan fingerprint with:

- radius: `2`;
- number of bits: `2048`;
- chirality information enabled.

### RDKit descriptors

The descriptor vector contains molecular properties such as:

- molecular weight;
- LogP;
- topological polar surface area;
- hydrogen-bond donors;
- hydrogen-bond acceptors;
- rotatable bonds;
- ring count;
- aromatic ring count;
- heavy-atom count;
- fraction Csp3;
- heteroatom count; and
- molecular refractivity.

The descriptor names, order, fingerprint configuration, and preprocessing steps must remain identical during training and backend inference.

---

## Processed Data Outputs

The data pipeline generates endpoint-specific processed files.

Typical outputs include:

```text
processed/admet/
├── endpoint_train.csv
├── endpoint_valid.csv
├── endpoint_test.csv
├── X_train.npy
├── X_valid.npy
├── X_test.npy
├── y_train.npy
├── y_valid.npy
├── y_test.npy
├── train_smiles.npy
├── valid_smiles.npy
├── test_smiles.npy
└── drug_id arrays
```

CSV files preserve readable molecule metadata, while NumPy arrays provide faster model loading and training.

---

## Models

Four machine-learning model families are trained for every endpoint:

- Extra Trees
- XGBoost
- CatBoost
- LightGBM

Each model family is trained using three random seeds:

```text
42, 52, 62
```

This gives:

```text
4 model families × 3 seeds = 12 trained models per endpoint
```

Across six endpoints, the complete prediction system uses:

```text
72 trained model files
```

Additional saved artifacts include endpoint-specific imputers and ensemble configurations.

---

## Seed Averaging

For a single model family, predictions from the three seeds are averaged:

```text
Family prediction
=
mean(
    seed 42 prediction,
    seed 52 prediction,
    seed 62 prediction
)
```

Seed averaging reduces sensitivity to randomness during training and produces more stable predictions.

---

## Ensemble Selection

After seed averaging, the four model-family outputs may be combined.

The ensemble prediction is:

```text
Final prediction
=
w₁ × Extra Trees
+
w₂ × XGBoost
+
w₃ × CatBoost
+
w₄ × LightGBM
```

where:

```text
w₁ + w₂ + w₃ + w₄ = 1
```

The ensemble weights are selected using validation-set performance only.

The held-out test set is evaluated only after the final model or ensemble has been selected.

### Final endpoint strategies

| Endpoint | Final Strategy |
|---|---|
| Absorption | CatBoost three-seed mean |
| Distribution | Equal-weight four-model ensemble |
| Metabolism | Equal-weight four-model ensemble |
| Excretion | Optimized weighted ensemble |
| Toxicity | Equal-weight four-model ensemble |
| Solubility | Optimized weighted ensemble |

### Endpoint-specific ensemble weights

| Endpoint | Extra Trees | XGBoost | CatBoost | LightGBM |
|---|---:|---:|---:|---:|
| Absorption | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Distribution | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Metabolism | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Excretion | 0.0000 | 0.2301 | 0.4287 | 0.3412 |
| Toxicity | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Solubility | 0.1907 | 0.0000 | 0.5303 | 0.2790 |

---

## Evaluation Metrics

Different metrics are used for regression and classification endpoints.

### Classification

Classification endpoints are evaluated using:

- **ROC-AUC**
- **AUPRC**

ROC-AUC measures ranking performance across classification thresholds.

AUPRC is especially useful when the positive and negative classes are imbalanced.

### Regression

Regression endpoints are evaluated using:

- **RMSE**
- **R²**

A lower RMSE indicates smaller prediction errors.

A higher R² indicates that the model explains a larger proportion of the variation in the target.

---

## Final Held-Out Test Results

| Endpoint | Test Samples | Selected Model | Primary Metric | Secondary Metric |
|---|---:|---|---:|---:|
| Absorption | 182 | CatBoost three-seed mean | RMSE: **0.4126** | R²: **0.6386** |
| Distribution | 406 | Equal-weight four-model ensemble | ROC-AUC: **0.9036** | AUPRC: **0.9675** |
| Metabolism | 2,467 | Equal-weight four-model ensemble | ROC-AUC: **0.8984** | AUPRC: **0.8766** |
| Excretion | 243 | Optimized weighted ensemble | RMSE: **44.5596** | R²: **0.1384** |
| Toxicity | 1,457 | Equal-weight four-model ensemble | ROC-AUC: **0.8322** | AUPRC: **0.8793** |
| Solubility | 1,998 | Optimized weighted ensemble | RMSE: **1.3293** | R²: **0.6497** |

These values are from the final held-out test sets and are not validation-set scores.

---

## Saved Model Artifacts

Each endpoint folder contains:

```text
endpoint/
├── imputer.joblib
├── ensemble_config.json
├── extra_trees_seed_42.joblib
├── extra_trees_seed_52.joblib
├── extra_trees_seed_62.joblib
├── xgboost_seed_42.joblib
├── xgboost_seed_52.joblib
├── xgboost_seed_62.joblib
├── catboost_seed_42.joblib
├── catboost_seed_52.joblib
├── catboost_seed_62.joblib
├── lightgbm_seed_42.joblib
├── lightgbm_seed_52.joblib
└── lightgbm_seed_62.joblib
```

Endpoint directories are mapped as follows:

```text
a → Absorption
d → Distribution
m → Metabolism
e → Excretion
t → Toxicity
s → Solubility
```

---

## Backend Inference

The FastAPI backend loads all endpoint artifacts during application startup.

For every submitted molecule, the backend:

1. validates the SMILES;
2. creates a canonical SMILES;
3. generates the 2,060-feature vector;
4. applies the endpoint-specific imputer;
5. obtains predictions from all seed models;
6. averages predictions for each model family;
7. combines family predictions using saved ensemble weights;
8. returns predictions and molecular descriptors; and
9. stores the request and response in SQLite.

### Classification output

For classification endpoints, the API returns a probability score.

### Regression output

For regression endpoints, the API returns the predicted continuous property value.

---

## Prediction History

Recent predictions are stored in:

```text
website/backend/data/prediction_history.db
```

The SQLite database stores information such as:

- original SMILES;
- canonical SMILES;
- inference mode;
- model version;
- molecular descriptors;
- endpoint predictions; and
- prediction timestamp.

Prediction history can be accessed through:

```text
GET /api/history
```

The history can be cleared using:

```text
DELETE /api/history
```

---

## Key Features

- SMILES validation and canonicalization
- Molecular structure visualization
- Six-endpoint ADMET and solubility prediction
- Morgan fingerprints and RDKit descriptors
- Scaffold-based splitting
- Separate held-out test evaluation
- Four machine-learning model families
- Three-seed model averaging
- Endpoint-specific weighted ensembles
- Interactive prediction cards
- Molecular descriptor panel
- Radar-chart visualization
- SQLite prediction history
- FastAPI REST API
- Swagger API documentation
- Responsive React frontend
- Cloudflare Pages deployment

---

## Technology Stack

### Machine Learning and Cheminformatics

- Python
- RDKit
- NumPy
- pandas
- scikit-learn
- XGBoost
- CatBoost
- LightGBM
- joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic
- SQLite

### Frontend

- React
- Vite
- Recharts
- Lucide React
- CSS

### Deployment

- Cloudflare Pages for the frontend
- Cloudflare Tunnel for the hackathon backend
- Docker configuration prepared for container deployment

---

## Project Structure

```text
kbg_hackathon/
├── data_pipeline/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── prepare_data.py
│   └── README.md
│
├── model_artifacts/
│   ├── a/
│   ├── d/
│   ├── e/
│   ├── m/
│   ├── s/
│   ├── t/
│   └── training_summary.json
│
├── models/
├── results/
├── saved_models/
│
├── website/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── chemistry.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── inference.py
│   │   │   ├── main.py
│   │   │   ├── model_loader.py
│   │   │   └── schemas.py
│   │   ├── data/
│   │   └── requirements.txt
│   │
│   └── frontend/
│       ├── public/
│       ├── src/
│       │   ├── components/
│       │   ├── data/
│       │   ├── services/
│       │   ├── App.jsx
│       │   ├── main.jsx
│       │   └── styles.css
│       ├── package.json
│       └── vite.config.js
│
├── deployment/
│   └── backend_space/
│       ├── Dockerfile
│       ├── README.md
│       └── .dockerignore
│
├── installed_packages.txt
├── requirements.txt
├── run_all.py
└── README.md
```

---

## Running Locally

### Prerequisites

- Python 3.11
- Node.js
- npm
- Git
- Git LFS for downloading large model artifacts

---

### 1. Clone the Repository

```bash
git clone https://github.com/dakshrathi-india/kbg_hackathon.git
cd kbg_hackathon
```

The completed application is available on the default `main` branch.

---

### 2. Install Git LFS

```bash
git lfs install
```

If the model artifacts are not present on the main branch, retrieve them from the model-artifact branch:

```bash
cd ..

git clone --depth 1 --filter=blob:none --sparse \
  --branch admet-models \
  https://github.com/dakshrathi-india/kbg_hackathon.git \
  temporary_models

cd temporary_models

git sparse-checkout set model_artifacts
git lfs pull

cp -r model_artifacts ../kbg_hackathon/model_artifacts

cd ..
rm -rf temporary_models
cd kbg_hackathon
```

Ensure the root structure is:

```text
model_artifacts/
├── a/
├── d/
├── e/
├── m/
├── s/
├── t/
└── training_summary.json
```

---

### 3. Create the Backend Environment

```bash
cd website/backend
python -m venv .venv
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 4. Configure the Backend

Create:

```text
website/backend/.env
```

Add:

```env
ALLOW_PLACEHOLDER_MODE=false
```

---

### 5. Start the Backend

```bash
uvicorn app.main:app --reload --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Model status:

```text
http://127.0.0.1:8000/api/models/status
```

The expected model status should show:

```text
loaded_endpoints: a, d, e, m, s, t
missing_endpoints: none
load_errors: none
```

---

### 6. Configure the Frontend

Create:

```text
website/frontend/.env
```

Add:

```env
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

### 7. Start the Frontend

Open a second terminal:

```bash
cd website/frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Basic API information |
| GET | `/api/health` | Backend, database, and model status |
| GET | `/api/models/status` | Loaded, missing, and failed endpoint models |
| POST | `/api/models/reload` | Reload saved model artifacts |
| POST | `/api/predict` | Predict all six molecular properties |
| GET | `/api/history` | Retrieve recent prediction history |
| DELETE | `/api/history` | Clear prediction history |

---

## Example Prediction Request

```http
POST /api/predict
Content-Type: application/json
```

```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

The example molecule is aspirin.

---

## Production Frontend Build

```bash
cd website/frontend
npm run build
```

The production build is generated inside:

```text
website/frontend/dist/
```

---

## Deployment Architecture

```text
Cloudflare Pages
      |
      | HTTPS
      v
Cloudflare Tunnel
      |
      v
FastAPI backend
      |
      v
Trained endpoint models
```

The frontend is deployed at:

```text
https://delta001-admet.pages.dev
```

The frontend deployment is persistent.

The hackathon backend currently depends on the availability of its configured FastAPI server and Cloudflare Tunnel. A permanent container-based backend deployment is planned.

---

## Limitations

- Predictions are computational estimates and should not replace laboratory experiments.
- Reliability depends on the chemical applicability domain of the training datasets.
- Molecules that differ substantially from the training data may receive less reliable predictions.
- The Excretion endpoint currently has lower predictive performance than the other endpoints.
- Scaffold splitting reduces structural leakage but creates a more difficult evaluation setting.
- The current backend deployment may depend on a temporary service or tunnel.
- SQLite history is local to the backend environment.
- The system does not yet provide calibrated prediction uncertainty.
- The application is intended for educational and preliminary screening purposes only.

---

## Future Improvements

- Permanent container-based backend deployment
- Applicability-domain detection
- Prediction uncertainty and confidence intervals
- Additional ADMET endpoints
- Batch SMILES prediction
- CSV upload and result export
- Molecular similarity search
- Feature-contribution explainability
- Hyperparameter optimization
- Model monitoring and versioning
- Graph neural network comparison
- Persistent cloud database integration

---

## Disclaimer

This project is intended for educational, research, and preliminary molecular-screening purposes only.

The predictions are not medical advice and must not be used as a substitute for:

- laboratory validation;
- toxicological testing;
- pharmacological experiments;
- clinical evaluation; or
- professional drug-development decisions.

---

## Team

**Team DELTA001**

KBG Computational Biology Hackathon  
Indian Institute of Technology Mandi

---

## Project Links

- **Live Demo:** https://delta001-admet.pages.dev
- **GitHub Repository:** https://github.com/dakshrathi-india/kbg_hackathon

---

⭐ Star the repository if you found the project useful.