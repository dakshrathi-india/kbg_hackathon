from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Molecular SMILES representation.",
    )