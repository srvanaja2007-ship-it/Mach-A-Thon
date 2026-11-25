"""Adapter for UrbanPulse's existing satellite fusion and inference pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

LEO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LEO_ROOT.parent
URBANPULSE_ROOT = PROJECT_ROOT / "migration & urbanization"
if str(URBANPULSE_ROOT) not in sys.path:
    sys.path.insert(0, str(URBANPULSE_ROOT))


def get_preset_hubs() -> dict[str, dict[str, Any]]:
    from data_pipeline import PRESET_HUBS
    return PRESET_HUBS


def analyze(lat: float, lon: float) -> dict[str, Any]:
    """Fetch/cache UrbanPulse feature layers, run its actual ML pipeline, summarize outputs."""
    from data_pipeline import fetch_location_data
    from ml_engine import ZONE_LABELS, run_ml_pipeline

    started = time.perf_counter()
    data = fetch_location_data(float(lat), float(lon))
    sources = data.get("sources", {})
    synthetic = [name for name, source in sources.items()
                 if "synthetic" in str(source).lower()]
    if synthetic:
        details = ", ".join(f"{name}={sources[name]}" for name in synthetic)
        raise RuntimeError(
            "UrbanPulse did not run because required layers resolved to synthetic fallback data: "
            f"{details}. Provide compatible real feature layers or restore the real-data cache."
        )

    result = run_ml_pipeline(data)
    zone_counts = {ZONE_LABELS[int(zone)]: int(count)
                   for zone, count in result["n_zones"].items()}
    return {
        "application": "UrbanPulse",
        "status": "SUCCESS",
        "location": {"lat": float(result["lat"]), "lon": float(result["lon"])},
        "grid": {"cells_per_side": int(result["n_cells"]),
                 "cell_size_m": int(result["grid_size_m"])},
        "data_sources": sources,
        "insights": {
            "zone_counts": zone_counts,
            "classifier_cv_accuracy": float(result["cv_accuracy"]),
            "new_built_area_sqkm": float(result["new_built_sqkm"]),
            "night_light_gap_alerts": int(result["n_nlg_alerts"]),
            "night_light_gap_area_sqkm": float(result["nlg_area_sqkm"]),
            "estimated_migrant_influx_2027_28": int(result["influx_2027_28"]),
            "sprawl_risk_mean": float(result["sprawl_prob_mean"]),
            "sprawl_risk_max": float(result["sprawl_prob_max"]),
            "feature_importances": {str(k): float(v)
                                    for k, v in result["feat_importances"].items()},
        },
        "processing_time_seconds": round(time.perf_counter() - started, 4),
    }

