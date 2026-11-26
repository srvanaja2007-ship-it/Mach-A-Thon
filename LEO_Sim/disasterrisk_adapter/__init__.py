"""Adapter for DisasterRisk's existing ClaimPulse scenario decision pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

LEO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LEO_ROOT.parent
DISASTERRISK_ROOT = PROJECT_ROOT / "DisasterRisk"
if str(DISASTERRISK_ROOT) not in sys.path:
    sys.path.insert(0, str(DISASTERRISK_ROOT))


def list_scenarios() -> dict[str, dict[str, Any]]:
    from backend.scenarios.data_builder import PRESET_EVENTS
    return PRESET_EVENTS


def analyze(preset_id: str) -> dict[str, Any]:
    """Execute DisasterRisk's registered connector, hazard, exposure, loss and decision modules."""
    from backend.core.schemas import EventRequest
    from backend.modules.implementations import (
        BaseHazardModule,
        DemoSatelliteConnector,
        MultiStakeholderDecisionModule,
        StandardExposureModule,
        TransparentEconomicModel,
    )

    presets = list_scenarios()
    if preset_id not in presets:
        raise ValueError(f"Unknown DisasterRisk scenario: {preset_id}")

    started = time.perf_counter()
    preset = presets[preset_id]
    request = EventRequest(preset_id=preset_id,
                           disaster_type=preset["disaster_type"],
                           intensity_multiplier=1.0)
    connector = DemoSatelliteConnector()
    raw = connector.fetch(request)
    # DemoSatelliteConnector always returns is_demo=True — we accept demo scenarios
    # and label the output accordingly so the dashboard is transparent about data quality.
    data_quality = "DEMO_SCENARIO" if raw.get("is_demo", True) else "OBSERVED_DATA"
    damage = BaseHazardModule(request.disaster_type, f"{request.disaster_type}-hazard").compute_damage(request, raw)
    exposure_result = StandardExposureModule().overlay_exposure(request, damage, raw)
    loss = TransparentEconomicModel().estimate_loss(request, damage, exposure_result, raw)
    decisions = MultiStakeholderDecisionModule().evaluate_decisions(
        request, damage, exposure_result, loss, raw,
    )
    hazard = damage["metrics"]
    exposure = exposure_result["summary"]
    return {
        "application": "DisasterRisk",
        "status": "SUCCESS",
        "data_status": data_quality,
        "data_quality": data_quality,
        # Representative satellite scene size: SAR+optical multi-spectral (1 MB per scene)
        "raw_input_bytes": 1_048_576,
        "event": {
            "id": raw["id"], "title": raw["title"],
            "disaster_type": raw["disaster_type"], "country": raw["country"],
            "state": raw["state"], "district": raw["district"],
            "start_date": raw["start_date"], "end_date": raw["end_date"],
            "center": raw["center"], "bbox": raw["bbox"],
        },
        "hazard": hazard.model_dump() if hasattr(hazard, "model_dump") else hazard,
        "exposure": exposure.model_dump() if hasattr(exposure, "model_dump") else exposure,
        "loss": {
            "total_expected_loss_inr": float(loss.total_expected_loss_inr),
            "business_interruption_inr": float(loss.business_interruption_inr),
            "uncertainty_low_inr": float(loss.uncertainty_low_inr),
            "uncertainty_high_inr": float(loss.uncertainty_high_inr),
            "by_sector": [item.model_dump() for item in loss.by_sector],
            "priority_zone_count": len(loss.top_10_zones),
        },
        "decisions": {
            "recommended_action_headline": decisions.recommended_action_headline,
            "overall_confidence": float(decisions.overall_confidence),
            "provisional_payout_inr": float(decisions.total_provisional_payout_inr),
            "recommended_relief_inr": float(decisions.total_recommended_relief_inr),
            "claims_fast_tracked_count": int(decisions.claims_fast_tracked_count),
            "claims_manual_review_count": int(decisions.claims_manual_review_count),
            "claims": [item.model_dump() for item in decisions.claims],
            "relief_allocations": [item.model_dump() for item in decisions.relief_allocations],
            "lender_alerts": [item.model_dump() for item in decisions.lender_alerts],
            "audit_trail": [item.model_dump() for item in decisions.audit_trail],
        },
        "processing_time_seconds": round(time.perf_counter() - started, 4),
    }

