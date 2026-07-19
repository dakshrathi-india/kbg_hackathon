from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.config import (
    create_required_directories,
    get_frontend_origins,
)

from app.database import (
    clear_prediction_history,
    get_recent_predictions,
    initialize_database,
    save_prediction,
)

from app.inference import (
    predict_molecule,
)

from app.model_loader import (
    model_registry,
)

from app.schemas import (
    PredictRequest,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    create_required_directories()
    initialize_database()
    model_registry.load_all()

    yield


app = FastAPI(
    title="DELTA001 ADMET API",
    description=(
        "SMILES-to-ADMET prediction backend "
        "for the DELTA001 hackathon project."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "message": "DELTA001 ADMET API",
        "documentation": "/docs",
    }


@app.get("/api/health")
def health() -> dict:
    status = model_registry.get_status()

    return {
        "status": "healthy",
        "database": "ready",
        "models": status,
    }


@app.get("/api/models/status")
def model_status() -> dict:
    return model_registry.get_status()


@app.post("/api/models/reload")
def reload_models() -> dict:
    model_registry.load_all()

    return {
        "message": "Model reload completed.",
        **model_registry.get_status(),
    }


@app.post("/api/predict")
def predict(
    request: PredictRequest,
) -> dict:
    try:
        result = predict_molecule(
            smiles=request.smiles,
            registry=model_registry,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{error}"
            ),
        ) from error

    history_id = save_prediction(
        original_smiles=(
            result["original_smiles"]
        ),
        canonical_smiles=(
            result["canonical_smiles"]
        ),
        inference_mode=(
            result["metadata"][
                "inference_mode"
            ]
        ),
        model_version=(
            result["metadata"][
                "model_version"
            ]
        ),
        descriptors=(
            result["descriptors"]
        ),
        predictions=(
            result["predictions"]
        ),
    )

    result["history_id"] = history_id

    return result


@app.get("/api/history")
def history(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
) -> dict:
    items = get_recent_predictions(
        limit=limit
    )

    return {
        "count": len(items),
        "items": items,
    }


@app.delete("/api/history")
def delete_history() -> dict:
    deleted_count = (
        clear_prediction_history()
    )

    return {
        "message": (
            "Prediction history cleared."
        ),
        "deleted_count": deleted_count,
    }