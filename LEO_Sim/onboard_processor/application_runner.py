"""Application dispatcher for independent LEO onboard model adapters."""

from __future__ import annotations

from typing import Any

APPLICATIONS = ("Eco_Act", "UrbanPulse", "DisasterRisk")


def run(application: str, **inputs: Any) -> dict[str, Any]:
    if application == "Eco_Act":
        from .processor import run as run_eco_act
        result = run_eco_act(inputs["image_path"])
        result["application"] = "Eco_Act"
        return result
    if application == "UrbanPulse":
        from urbanpulse_adapter import analyze
        return analyze(inputs["lat"], inputs["lon"])
    if application == "DisasterRisk":
        from disasterrisk_adapter import analyze
        return analyze(inputs["preset_id"])
    raise ValueError(f"Unsupported onboard application: {application}")

