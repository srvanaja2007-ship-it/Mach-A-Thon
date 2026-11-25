"""Bridge connecting Track B verified edge/aerial alerts directly into Track A (ClaimPulse)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

_DISASTER_RISK_ROOT = str(Path(__file__).resolve().parent.parent)
if _DISASTER_RISK_ROOT not in sys.path:
    sys.path.insert(0, _DISASTER_RISK_ROOT)

from backend.core.schemas import BoundingBox, EventFullResponse, EventRequest  # noqa: E402
from backend.main import ingest_track_b_alert  # noqa: E402
from backend.modules.implementations import INR_SCALE  # noqa: E402
from track_b.aerial_dispatch import AerialVerificationReport  # noqa: E402
from track_b.schema import TrackBEdgeAlert  # noqa: E402


@dataclass
class TrackABridgeResult:
    """Structured economic and operational decision output returned by Track A."""

    event_id: str
    executed_action_code: str
    parametric_trigger_status: str
    downlink_priority: str
    total_expected_loss_inr: float
    total_expected_loss_usd: float
    total_provisional_payout_inr: float
    total_provisional_payout_usd: float
    total_recommended_relief_inr: float
    total_recommended_relief_usd: float
    claims_fast_tracked_count: int
    claims_manual_review_count: int
    overall_confidence: float
    top_priority_zone: str
    recommended_action_headline: str
    audit_records_count: int
    latest_audit_action: str
    track_a_bundle: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def track_b_alert_to_event_request(
    alert: TrackBEdgeAlert,
    *,
    title: str | None = None,
    country: str = "India",
    state: str = "Odisha",
    district: str = "Coastal Flood Corridor",
) -> EventRequest:
    """Convert a validated TrackBEdgeAlert into Track A's EventRequest contract."""
    min_lon, min_lat, max_lon, max_lat = alert.bounding_box
    scene_date = alert.scene_timestamp_utc[:10]
    return EventRequest(
        preset_id=None,
        disaster_type=alert.detected_peril,
        title=title or f"Track B {alert.satellite_id} {alert.detected_peril.title()} Alert — {district}",
        country=country,
        state=state,
        district=district,
        start_date=scene_date,
        end_date=scene_date,
        bbox=BoundingBox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
        ),
        intensity_multiplier=1.0,
    )


def ingest_track_b_into_track_a(
    alert: TrackBEdgeAlert,
    aerial_report: AerialVerificationReport | None = None,
    *,
    title: str | None = None,
    country: str = "India",
    state: str = "Odisha",
    district: str = "Coastal Flood Corridor",
) -> TrackABridgeResult:
    """Feed Track B verified alert into Track A and execute financial loss & payout triage."""
    alert_dict = alert.model_dump()
    aerial_dict = aerial_report.model_dump() if aerial_report is not None else None

    bundle: EventFullResponse = ingest_track_b_alert(
        alert_dict,
        aerial_verification=aerial_dict,
        title=title,
        country=country,
        state=state,
        district=district,
    )

    loss_inr = float(bundle.loss.total_expected_loss_inr)
    payout_inr = float(bundle.decisions.total_provisional_payout_inr)
    relief_inr = float(bundle.decisions.total_recommended_relief_inr)
    top_zone = bundle.loss.top_10_zones[0].zone_name if bundle.loss.top_10_zones else "N/A"
    latest_audit = bundle.decisions.audit_trail[0].action if bundle.decisions.audit_trail else "N/A"

    return TrackABridgeResult(
        event_id=bundle.event.id,
        executed_action_code=alert.recommended_action_code,
        parametric_trigger_status=alert.parametric_trigger_status,
        downlink_priority=alert.downlink_priority,
        total_expected_loss_inr=round(loss_inr, 2),
        total_expected_loss_usd=round(loss_inr / INR_SCALE, 2),
        total_provisional_payout_inr=round(payout_inr, 2),
        total_provisional_payout_usd=round(payout_inr / INR_SCALE, 2),
        total_recommended_relief_inr=round(relief_inr, 2),
        total_recommended_relief_usd=round(relief_inr / INR_SCALE, 2),
        claims_fast_tracked_count=bundle.decisions.claims_fast_tracked_count,
        claims_manual_review_count=bundle.decisions.claims_manual_review_count,
        overall_confidence=bundle.decisions.overall_confidence,
        top_priority_zone=top_zone,
        recommended_action_headline=bundle.decisions.recommended_action_headline,
        audit_records_count=len(bundle.decisions.audit_trail),
        latest_audit_action=latest_audit,
        track_a_bundle=bundle.model_dump(),
    )
