"""Application dispatcher for independent LEO onboard model adapters."""

from __future__ import annotations

import math
import os
from typing import Any

APPLICATIONS = {"Eco_Act", "UrbanPulse", "DisasterRisk"}


def run(application: str, **inputs: Any) -> dict[str, Any]:
    if application not in APPLICATIONS:
        raise ValueError(f"Unsupported onboard application: {application}")
    if application == "Eco_Act":
        image_path = inputs.get("image_path")
        if not isinstance(image_path, (str, bytes, os.PathLike)) or not image_path:
            raise ValueError("Eco_Act requires an image_path")
        image_path = os.fspath(image_path)
        from .processor import run as run_eco_act
        result = run_eco_act(image_path)
        result["application"] = "Eco_Act"
        return result
    if application == "UrbanPulse":
        try:
            lat, lon = float(inputs["lat"]), float(inputs["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("UrbanPulse requires numeric lat and lon") from exc
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("UrbanPulse coordinates must be finite WGS84 values")
        from urbanpulse_adapter import analyze
        return analyze(lat, lon)
    if application == "DisasterRisk":
        preset_id = inputs.get("preset_id")
        if not isinstance(preset_id, str) or not preset_id.strip():
            raise ValueError("DisasterRisk requires a preset_id")
        from disasterrisk_adapter import analyze
        return analyze(preset_id)

