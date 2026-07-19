import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import (
    ENDPOINT_CONFIG,
    MODEL_ARTIFACTS_DIR,
)


MODEL_PATTERNS = {
    "extra_trees": [
        "extra_trees_seed_*.joblib",
        "extra_trees.joblib",
    ],
    "xgboost": [
        "xgboost_seed_*.joblib",
        "xgboost.joblib",
    ],
    "catboost": [
        "catboost_seed_*.joblib",
        "catboost.joblib",
    ],
    "lightgbm": [
        "lightgbm_seed_*.joblib",
        "lightgbm.joblib",
    ],
}


@dataclass
class EndpointBundle:
    endpoint: str
    task_type: str
    imputer: Any
    models: dict[str, list[Any]]
    weights: dict[str, float]
    model_version: str


class ModelRegistry:
    def __init__(self) -> None:
        self.bundles: dict[
            str,
            EndpointBundle,
        ] = {}

        self.load_errors: dict[
            str,
            str,
        ] = {}

    def load_all(self) -> None:
        self.bundles.clear()
        self.load_errors.clear()

        for endpoint in ENDPOINT_CONFIG:
            try:
                bundle = self._load_endpoint(
                    endpoint
                )

                self.bundles[
                    endpoint
                ] = bundle

            except Exception as error:
                self.load_errors[
                    endpoint
                ] = str(error)

    def _load_endpoint(
        self,
        endpoint: str,
    ) -> EndpointBundle:
        endpoint_directory = (
            MODEL_ARTIFACTS_DIR
            / endpoint
        )

        imputer_path = (
            endpoint_directory
            / "imputer.joblib"
        )

        if not imputer_path.exists():
            raise FileNotFoundError(
                f"Missing {imputer_path.name}"
            )

        imputer = joblib.load(
            imputer_path
        )

        models: dict[str, list[Any]] = {}

        for (
            algorithm,
            patterns,
        ) in MODEL_PATTERNS.items():
            model_paths: list[Path] = []

            for pattern in patterns:
                model_paths.extend(
                    endpoint_directory.glob(
                        pattern
                    )
                )

            unique_paths = sorted(
                set(model_paths)
            )

            if unique_paths:
                models[algorithm] = [
                    joblib.load(path)
                    for path in unique_paths
                ]

        if not models:
            raise FileNotFoundError(
                "No trained model files found."
            )

        config_path = (
            endpoint_directory
            / "ensemble_config.json"
        )

        configuration = {}

        if config_path.exists():
            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as file:
                configuration = json.load(
                    file
                )

        weights = configuration.get(
            "algorithm_weights",
            {},
        )

        available_weights = {
            algorithm: float(
                weights.get(algorithm, 0.0)
            )
            for algorithm in models
        }

        total_weight = sum(
            available_weights.values()
        )

        if total_weight <= 0:
            equal_weight = (
                1.0 / len(models)
            )

            available_weights = {
                algorithm: equal_weight
                for algorithm in models
            }

        else:
            available_weights = {
                algorithm: weight
                / total_weight
                for algorithm, weight
                in available_weights.items()
            }

        return EndpointBundle(
            endpoint=endpoint,
            task_type=ENDPOINT_CONFIG[
                endpoint
            ]["task_type"],
            imputer=imputer,
            models=models,
            weights=available_weights,
            model_version=configuration.get(
                "model_version",
                "final-ensemble-v1",
            ),
        )

    def is_ready(
        self,
        endpoint: str,
    ) -> bool:
        return endpoint in self.bundles

    def predict_endpoint(
        self,
        endpoint: str,
        feature_vector: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        if endpoint not in self.bundles:
            raise RuntimeError(
                f"Models for endpoint "
                f"{endpoint.upper()} are not loaded."
            )

        bundle = self.bundles[
            endpoint
        ]

        expected_features = getattr(
            bundle.imputer,
            "n_features_in_",
            None,
        )

        if (
            expected_features is not None
            and feature_vector.shape[0]
            != expected_features
        ):
            raise ValueError(
                f"Feature mismatch for "
                f"{endpoint.upper()}: backend generated "
                f"{feature_vector.shape[0]} features, "
                f"but the imputer expects "
                f"{expected_features}."
            )

        features = feature_vector.reshape(
            1,
            -1,
        )

        transformed_features = (
            bundle.imputer.transform(
                features
            )
        )

        algorithm_predictions = {}

        for (
            algorithm,
            algorithm_models,
        ) in bundle.models.items():
            seed_predictions = []

            for model in algorithm_models:
                prediction = (
                    self._predict_single_model(
                        model=model,
                        features=(
                            transformed_features
                        ),
                        task_type=(
                            bundle.task_type
                        ),
                    )
                )

                seed_predictions.append(
                    prediction
                )

            algorithm_predictions[
                algorithm
            ] = float(
                np.mean(seed_predictions)
            )

        final_prediction = 0.0

        for (
            algorithm,
            prediction,
        ) in algorithm_predictions.items():
            final_prediction += (
                bundle.weights.get(
                    algorithm,
                    0.0,
                )
                * prediction
            )

        return (
            float(final_prediction),
            algorithm_predictions,
        )

    @staticmethod
    def _predict_single_model(
        model: Any,
        features: np.ndarray,
        task_type: str,
    ) -> float:
        if task_type == "classification":
            if hasattr(
                model,
                "predict_proba",
            ):
                probabilities = (
                    model.predict_proba(
                        features
                    )
                )

                return float(
                    probabilities[0][1]
                )

        prediction = model.predict(
            features
        )

        return float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

    def get_status(self) -> dict:
        return {
            "loaded_endpoints": sorted(
                self.bundles.keys()
            ),
            "missing_endpoints": sorted(
                self.load_errors.keys()
            ),
            "load_errors": (
                self.load_errors
            ),
        }


model_registry = ModelRegistry()