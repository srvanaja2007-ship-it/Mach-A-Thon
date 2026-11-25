"""Safe integration bridge for Model 1 satellite geolocation outputs.

The existing correction models were trained on GSDC GNSS kinematic features.
They can only be used for another system when Model 1 supplies a feature
vector with that exact schema. Image pixels or Eco_Act economic features are
not silently substituted for GNSS features.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MAX_PHYSICAL_CORRECTION_METERS, MODELS_DIR
from src.model import GpsErrorCorrectionModel


@dataclass
class SatelliteCorrectionResult:
    image_id: Optional[str]
    original_latitude: float
    original_longitude: float
    random_forest: Optional[dict[str, float]]
    xgboost: Optional[dict[str, float]]
    selected_model: Optional[str]
    final_latitude: Optional[float]
    final_longitude: Optional[float]
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    lat = float(latitude)
    lon = float(longitude)
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError("Predicted latitude and longitude must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Predicted latitude is outside [-90, 90]: {lat}")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Predicted longitude is outside [-180, 180]: {lon}")
    return lat, lon


def _build_feature_frame(features: Optional[Mapping[str, Any]]) -> Optional[pd.DataFrame]:
    """Build one ordered inference row, rejecting missing values."""
    if features is None or not set(FEATURE_COLUMNS).issubset(features):
        return None
    frame = pd.DataFrame(
        [{name: features[name] for name in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Model 1 feature values must be finite numeric values.")
    return numeric


def _predict(model, lat: float, lon: float, features: pd.DataFrame, max_m: Optional[float]):
    lat_corr, lon_corr, pred_n, pred_e = model.correct_coordinates(
        np.array([lat], dtype=float), np.array([lon], dtype=float), features,
        max_correction_meters=max_m,
    )
    return {
        "latitude": float(lat_corr[0]),
        "longitude": float(lon_corr[0]),
        "correction_north_m": float(pred_n[0]),
        "correction_east_m": float(pred_e[0]),
    }


def correct_satellite_image_location(
    image: Any,
    predicted_latitude: float,
    predicted_longitude: float,
    image_features: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    model_dir: Path = MODELS_DIR,
    selected_model: str = "random_forest",
    max_correction_meters: Optional[float] = MAX_PHYSICAL_CORRECTION_METERS,
) -> dict[str, Any]:
    """Run independent corrections on a Model 1 coordinate.

    The current GNSS models do not consume image pixels. If ``image_features``
    does not contain the exact causal GSDC feature schema, this function
    returns ``insufficient_features`` and leaves the coordinate unchanged.
    """
    lat, lon = _validate_coordinates(predicted_latitude, predicted_longitude)
    image_id = metadata.get("image_id") if isinstance(metadata, Mapping) else None
    selected_key = selected_model.lower().strip().replace(" ", "_")
    if selected_key in {"rf", "randomforest"}:
        selected_key = "random_forest"
    elif selected_key in {"xgb"}:
        selected_key = "xgboost"
    if selected_key not in {"random_forest", "xgboost"}:
        raise ValueError("selected_model must be 'random_forest' or 'xgboost'.")

    features = _build_feature_frame(image_features)
    if features is None:
        return SatelliteCorrectionResult(
            image_id=str(image_id) if image_id is not None else None,
            original_latitude=lat,
            original_longitude=lon,
            random_forest=None,
            xgboost=None,
            selected_model=None,
            final_latitude=None,
            final_longitude=None,
            status="insufficient_features",
            message=(
                "No correction applied: the existing models require the GSDC "
                "GNSS feature schema. Eco_Act image/economic features are not "
                "compatible; train a satellite-localization model first."
            ),
        ).to_dict()

    # ``image`` is intentionally not used: these models were not trained on pixels.
    del image
    rf_result = _predict(
        GpsErrorCorrectionModel.load(load_dir=model_dir, prefix="random_forest"),
        lat, lon, features, max_correction_meters,
    )
    xgb_result = _predict(
        GpsErrorCorrectionModel.load(load_dir=model_dir, prefix="xgboost"),
        lat, lon, features, max_correction_meters,
    )
    chosen = rf_result if selected_key == "random_forest" else xgb_result
    return SatelliteCorrectionResult(
        image_id=str(image_id) if image_id is not None else None,
        original_latitude=lat,
        original_longitude=lon,
        random_forest=rf_result,
        xgboost=xgb_result,
        selected_model=selected_key,
        final_latitude=chosen["latitude"],
        final_longitude=chosen["longitude"],
        status="corrected",
        message="RF and XGBoost ran independently on the same feature row.",
    ).to_dict()
