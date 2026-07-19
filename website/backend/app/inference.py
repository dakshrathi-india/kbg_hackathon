import hashlib
from typing import Any

from app.chemistry import (
    build_display_descriptors,
    build_model_features,
    parse_smiles,
    render_molecule_svg,
)

from app.config import (
    ALLOW_PLACEHOLDER_MODE,
    ENDPOINT_CONFIG,
)

from app.model_loader import (
    ModelRegistry,
)


def deterministic_number(
    canonical_smiles: str,
    endpoint: str,
) -> float:
    text = (
        f"{canonical_smiles}:{endpoint}"
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    integer = int(
        digest[:12],
        16,
    )

    return (
        integer
        / float(16**12 - 1)
    )


def placeholder_prediction(
    canonical_smiles: str,
    endpoint: str,
) -> float:
    random_value = deterministic_number(
        canonical_smiles,
        endpoint,
    )

    if endpoint == "a":
        return -6.5 + (
            random_value * 3.0
        )

    if endpoint == "e":
        return 10.0 + (
            random_value * 85.0
        )

    if endpoint == "s":
        return -6.0 + (
            random_value * 5.0
        )

    return 0.08 + (
        random_value * 0.84
    )


def get_prediction_label(
    endpoint: str,
    value: float,
) -> str:
    if endpoint == "a":
        return (
            "Higher predicted permeability"
            if value > -5
            else "Lower predicted permeability"
        )

    if endpoint == "d":
        return (
            "Likely BBB permeable"
            if value >= 0.5
            else "Likely BBB non-permeable"
        )

    if endpoint == "m":
        return (
            "Higher inhibition likelihood"
            if value >= 0.5
            else "Lower inhibition likelihood"
        )

    if endpoint == "e":
        return (
            "Higher predicted clearance"
            if value > 50
            else "Lower predicted clearance"
        )

    if endpoint == "t":
        return (
            "Higher mutagenicity likelihood"
            if value >= 0.5
            else "Lower mutagenicity likelihood"
        )

    if endpoint == "s":
        return (
            "Relatively higher aqueous solubility"
            if value > -3
            else "Relatively lower aqueous solubility"
        )

    return "Prediction generated"


def predict_molecule(
    smiles: str,
    registry: ModelRegistry,
) -> dict[str, Any]:
    molecule, canonical_smiles = (
        parse_smiles(smiles)
    )

    descriptors = (
        build_display_descriptors(
            molecule
        )
    )

    feature_vector = (
        build_model_features(
            molecule
        )
    )

    molecule_svg = (
        render_molecule_svg(
            molecule
        )
    )

    predictions = {}

    real_endpoint_count = 0
    placeholder_endpoint_count = 0
    model_versions = set()

    endpoint_key_names = {
        "a": "absorption",
        "d": "distribution",
        "m": "metabolism",
        "e": "excretion",
        "t": "toxicity",
        "s": "solubility",
    }

    for (
        endpoint,
        endpoint_information,
    ) in ENDPOINT_CONFIG.items():
        algorithm_predictions = {}

        if registry.is_ready(endpoint):
            value, algorithm_predictions = (
                registry.predict_endpoint(
                    endpoint=endpoint,
                    feature_vector=(
                        feature_vector
                    ),
                )
            )

            bundle = registry.bundles[
                endpoint
            ]

            model_versions.add(
                bundle.model_version
            )

            model_name = (
                "Four-model seed-averaged ensemble"
            )

            real_endpoint_count += 1

        else:
            if not ALLOW_PLACEHOLDER_MODE:
                raise RuntimeError(
                    f"Models for endpoint "
                    f"{endpoint.upper()} "
                    f"are unavailable."
                )

            value = placeholder_prediction(
                canonical_smiles,
                endpoint,
            )

            model_name = (
                "Development placeholder"
            )

            placeholder_endpoint_count += 1

        task_type = endpoint_information[
            "task_type"
        ]

        if task_type == "classification":
            value = max(
                0.0,
                min(1.0, value),
            )

        prediction_key = (
            endpoint_key_names[
                endpoint
            ]
        )

        predictions[
            prediction_key
        ] = {
            "code": endpoint.upper(),
            "endpoint": (
                endpoint_information[
                    "name"
                ]
            ),
            "property": (
                endpoint_information[
                    "property"
                ]
            ),
            "type": task_type,
            "value": round(
                float(value),
                6,
            ),
            "unit": (
                endpoint_information[
                    "unit"
                ]
            ),
            "label": get_prediction_label(
                endpoint,
                float(value),
            ),
            "model": model_name,
            "algorithm_predictions": {
                algorithm: round(
                    prediction,
                    6,
                )
                for (
                    algorithm,
                    prediction,
                ) in (
                    algorithm_predictions.items()
                )
            },
        }

    if real_endpoint_count == len(
        ENDPOINT_CONFIG
    ):
        inference_mode = "real"

    elif real_endpoint_count > 0:
        inference_mode = "hybrid"

    else:
        inference_mode = "placeholder"

    model_version = (
        ", ".join(
            sorted(model_versions)
        )
        if model_versions
        else "development-placeholder"
    )

    return {
        "valid": True,
        "original_smiles": smiles.strip(),
        "canonical_smiles": (
            canonical_smiles
        ),
        "molecule_svg": molecule_svg,
        "descriptors": descriptors,
        "predictions": predictions,
        "metadata": {
            "inference_mode": (
                inference_mode
            ),
            "model_version": (
                model_version
            ),
            "real_endpoint_count": (
                real_endpoint_count
            ),
            "placeholder_endpoint_count": (
                placeholder_endpoint_count
            ),
            "feature_count": int(
                feature_vector.shape[0]
            ),
            "disclaimer": (
                "Research prototype for "
                "preliminary computational "
                "screening only."
            ),
        },
    }