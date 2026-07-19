# DELTA001 — SMILES to ADMET Prediction

A machine-learning web application that predicts important **ADMET properties** and **aqueous solubility** directly from a molecular **SMILES** string.

Developed by **Team DELTA001** for the **KBG Computational Biology Hackathon, IIT Mandi**.

## Live Project

- **Deployed Application:** https://delta001-admet.pages.dev
- **GitHub Repository:** https://github.com/dakshrathi-india/kbg_hackathon

## Overview

A molecule can show promising biological activity and still fail as a drug because of poor absorption, distribution, metabolism, excretion, toxicity, or solubility.

DELTA001 provides a fast preliminary screening system that accepts a SMILES string and predicts six molecular properties:

| Code | Category | Predicted Property | Dataset | Task |
|---|---|---|---|---|
| A | Absorption | Caco-2 Permeability | Caco2_Wang | Regression |
| D | Distribution | Blood-Brain Barrier Penetration | BBB_Martins | Classification |
| M | Metabolism | CYP3A4 Inhibition | CYP3A4_Veith | Classification |
| E | Excretion | Hepatocyte Clearance | Clearance_Hepatocyte_AZ | Regression |
| T | Toxicity | AMES Mutagenicity | AMES | Classification |
| S | Solubility | Aqueous Solubility | Solubility_AqSolDB | Regression |

The application validates the molecule, generates molecular features, performs ensemble inference, displays the molecular structure and descriptors, and stores recent predictions.

## Final Held-Out Test Results

| Endpoint | Test Samples | Selected Model | Primary Metric | Secondary Metric |
|---|---:|---|---:|---:|
| Absorption | 182 | CatBoost, three-seed mean | RMSE: **0.4126** | R²: **0.6386** |
| Distribution | 406 | Equal-weight four-model ensemble | ROC-AUC: **0.9036** | AUPRC: **0.9675** |
| Metabolism | 2,467 | Equal-weight four-model ensemble | ROC-AUC: **0.8984** | AUPRC: **0.8766** |
| Excretion | 243 | Optimized weighted ensemble | RMSE: **44.5596** | R²: **0.1384** |
| Toxicity | 1,457 | Equal-weight four-model ensemble | ROC-AUC: **0.8322** | AUPRC: **0.8793** |
| Solubility | 1,998 | Optimized weighted ensemble | RMSE: **1.3293** | R²: **0.6497** |

These values are from the final held-out test sets.

## Key Features

- SMILES validation and canonicalization
- Molecular structure visualization
- Six-property ADMET and solubility prediction
- Morgan fingerprints and RDKit descriptors
- Four machine-learning model families
- Three-seed model averaging
- Endpoint-specific weighted ensembles
- Interactive prediction cards
- Molecular descriptor panel
- Radar-chart visualization
- SQLite prediction history
- FastAPI REST API
- Responsive React frontend

## Molecular Representation

Each molecule is represented using:

- **2,048-bit Morgan fingerprint**
- **12 RDKit molecular descriptors**
- **2,060 total input features**

The descriptors include molecular weight, LogP, TPSA, hydrogen-bond donors, hydrogen-bond acceptors, rotatable bonds, ring information, heavy-atom count, fraction Csp3, heteroatom count, and molecular refractivity.

## Data Processing Pipeline

The data pipeline performs:

1. Invalid-SMILES removal
2. SMILES canonicalization
3. Exact-duplicate removal
4. Missing-label checks
5. Conflicting duplicate-label checks
6. Scaffold-based train-validation splitting
7. Morgan fingerprint generation
8. RDKit descriptor calculation
9. Endpoint-wise feature and label export

A scaffold split is used to reduce structural leakage and provide a more realistic estimate of performance on unseen molecular scaffolds.

## Models

Four model families are trained for every endpoint:

- Extra Trees
- XGBoost
- CatBoost
- LightGBM

Each model family is trained using three random seeds:

```text
42, 52, 62
```

Predictions are first averaged across seeds. The model-family outputs are then combined using equal or optimized endpoint-specific ensemble weights.

## System Architecture

```text
SMILES Input
     |
     v
React + Vite Frontend
     |
     v
FastAPI Backend
     |
     +--> SMILES validation
     |
     +--> Canonicalization
     |
     +--> Morgan fingerprint generation
     |
     +--> RDKit descriptor calculation
     |
     +--> Missing-value imputation
     |
     +--> Four model families × three seeds
     |
     +--> Seed averaging
     |
     +--> Endpoint-specific ensemble
     |
     v
ADMET + Solubility Predictions
     |
     +--> Molecular structure
     +--> Molecular descriptors
     +--> Prediction history
```

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
- FastAPI backend exposed through a configured backend service/tunnel during the hackathon deployment

## Project Structure

```text
kbg_hackathon/
├── data_pipeline/
│   ├── data_loader.py
│   ├── prepare_data.py
│   └── README.md
├── model_artifacts/
│   ├── a/
│   ├── d/
│   ├── e/
│   ├── m/
│   ├── s/
│   ├── t/
│   └── training_summary.json
├── models/
├── results/
├── saved_models/
├── website/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── chemistry.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── inference.py
│   │   │   ├── main.py
│   │   │   ├── model_loader.py
│   │   │   └── schemas.py
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/
│       ├── package.json
│       └── vite.config.js
├── deployment/
│   └── backend_space/
├── requirements.txt
└── README.md
```

Large model artifacts may be excluded from normal Git history because of their size. Place the required endpoint folders inside `model_artifacts/` before starting the real-inference backend.

## Running Locally

### Prerequisites

- Python 3.11
- Node.js
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/dakshrathi-india/kbg_hackathon.git
cd kbg_hackathon
git checkout website-setup
```

### 2. Prepare Model Artifacts

Ensure the following structure exists at the repository root:

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

Each endpoint directory should contain:

- `imputer.joblib`
- `ensemble_config.json`
- Extra Trees models for seeds 42, 52 and 62
- XGBoost models for seeds 42, 52 and 62
- CatBoost models for seeds 42, 52 and 62
- LightGBM models for seeds 42, 52 and 62

### 3. Start the Backend

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

Start FastAPI:

```bash
uvicorn app.main:app --reload --port 8000
```

Backend documentation:

```text
http://127.0.0.1:8000/docs
```

Model status:

```text
http://127.0.0.1:8000/api/models/status
```

### 4. Configure the Frontend

Create `website/frontend/.env`:

```env
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 5. Start the Frontend

```bash
cd website/frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/api/health` | Backend, database, and model status |
| GET | `/api/models/status` | Loaded and missing endpoint models |
| POST | `/api/models/reload` | Reload model artifacts |
| POST | `/api/predict` | Predict all six molecular properties |
| GET | `/api/history` | Retrieve prediction history |
| DELETE | `/api/history` | Clear prediction history |

### Example Prediction Request

```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

The example molecule is aspirin.

## Production Frontend Build

```bash
cd website/frontend
npm run build
```

The production build is generated inside:

```text
website/frontend/dist/
```

## Limitations

- Predictions are computational estimates and should not replace laboratory experiments.
- Performance depends on the chemical applicability domain of each training dataset.
- Excretion currently has lower predictive performance than the other endpoints.
- Scaffold splitting reduces leakage but makes the evaluation more challenging.
- The hackathon backend deployment may depend on the availability of its configured backend service.
- The tool is intended for educational and preliminary screening purposes only.

## Future Improvements

- Permanent cloud deployment of the inference backend
- Applicability-domain estimation
- Prediction-confidence and uncertainty scores
- Additional ADMET endpoints
- Batch CSV and multi-SMILES prediction
- Explainability using descriptor and fingerprint contributions
- Hyperparameter optimization
- Model versioning and monitoring
- Comparison with graph neural networks

## Disclaimer

This project is intended for educational, research, and preliminary screening purposes only. The predictions are not medical advice and must not be used as a substitute for laboratory validation, toxicological testing, or professional drug-development decisions.

## Team

**DELTA001**

KBG Computational Biology Hackathon  
Indian Institute of Technology Mandi

---

⭐ **GitHub:** https://github.com/dakshrathi-india/kbg_hackathon  
🚀 **Live Demo:** https://delta001-admet.pages.dev
