import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=20,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                original_smiles TEXT NOT NULL,
                canonical_smiles TEXT NOT NULL,
                inference_mode TEXT NOT NULL,
                model_version TEXT NOT NULL,
                descriptors_json TEXT NOT NULL,
                predictions_json TEXT NOT NULL
            )
            """
        )

        connection.commit()


def save_prediction(
    original_smiles: str,
    canonical_smiles: str,
    inference_mode: str,
    model_version: str,
    descriptors: dict[str, Any],
    predictions: dict[str, Any],
) -> int:
    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at,
                original_smiles,
                canonical_smiles,
                inference_mode,
                model_version,
                descriptors_json,
                predictions_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                original_smiles,
                canonical_smiles,
                inference_mode,
                model_version,
                json.dumps(descriptors),
                json.dumps(predictions),
            ),
        )

        connection.commit()

        return int(cursor.lastrowid)


def get_recent_predictions(
    limit: int = 10,
) -> list[dict[str, Any]]:
    safe_limit = max(
        1,
        min(limit, 50),
    )

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                original_smiles,
                canonical_smiles,
                inference_mode,
                model_version,
                descriptors_json,
                predictions_json
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    history = []

    for row in rows:
        history.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "original_smiles": (
                    row["original_smiles"]
                ),
                "canonical_smiles": (
                    row["canonical_smiles"]
                ),
                "inference_mode": (
                    row["inference_mode"]
                ),
                "model_version": (
                    row["model_version"]
                ),
                "descriptors": json.loads(
                    row["descriptors_json"]
                ),
                "predictions": json.loads(
                    row["predictions_json"]
                ),
            }
        )

    return history


def clear_prediction_history() -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM predictions"
        )

        connection.commit()

        return int(cursor.rowcount)