import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


# kbg_hack_ps1/website/backend
BACKEND_DIR = Path(__file__).resolve().parent.parent

# kbg_hack_ps1
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Runtime database remains inside website/backend/data
DATA_DIR = BACKEND_DIR / "data"
DATABASE_PATH = DATA_DIR / "prediction_history.db"

# Final trained models are stored at kbg_hack_ps1/model_artifacts
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"


ENDPOINT_CONFIG = {
    "a": {
        "name": "Absorption",
        "property": "Caco-2 Permeability",
        "task_type": "regression",
        "unit": "log permeability",
    },
    "d": {
        "name": "Distribution",
        "property": "BBB Penetration",
        "task_type": "classification",
        "unit": "probability",
    },
    "m": {
        "name": "Metabolism",
        "property": "CYP3A4 Inhibition",
        "task_type": "classification",
        "unit": "probability",
    },
    "e": {
        "name": "Excretion",
        "property": "Hepatocyte Clearance",
        "task_type": "regression",
        "unit": "dataset-specific clearance unit",
    },
    "t": {
        "name": "Toxicity",
        "property": "AMES Mutagenicity",
        "task_type": "classification",
        "unit": "probability",
    },
    "s": {
        "name": "Solubility",
        "property": "Aqueous Solubility",
        "task_type": "regression",
        "unit": "log mol/L",
    },
}


def get_frontend_origins() -> list[str]:
    value = os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )

    return [
        origin.strip()
        for origin in value.split(",")
        if origin.strip()
    ]


ALLOW_PLACEHOLDER_MODE = (
    os.getenv("ALLOW_PLACEHOLDER_MODE", "true").lower()
    == "true"
)


def create_required_directories() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for endpoint in ENDPOINT_CONFIG:
        (
            MODEL_ARTIFACTS_DIR / endpoint
        ).mkdir(
            parents=True,
            exist_ok=True,
        )