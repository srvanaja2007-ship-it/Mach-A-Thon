"""Track B ("Build the Orbit") — Edge Satellite AI, 3-Tier Aerial Verification & Bridge."""
from __future__ import annotations

from track_b.aerial_dispatch import (
    AerialVerificationReport,
    DroneDispatchTelemetry,
    HapsDispatchTelemetry,
    SatelliteGnssTelemetry,
    run_aerial_verification,
)
from track_b.bridge import (
    TrackABridgeResult,
    ingest_track_b_into_track_a,
    track_b_alert_to_event_request,
)
from track_b.edge_inference import (
    RAW_SCENE_BYTES,
    RAW_SCENE_MB,
    EdgeInferenceResult,
    run_edge_satellite_inference,
)
from track_b.schema import (
    InferenceBenchmark,
    TrackBEdgeAlert,
    TrackBMergedTelemetry,
    using_pydantic_v2,
)

__all__ = [
    "AerialVerificationReport",
    "DroneDispatchTelemetry",
    "EdgeInferenceResult",
    "HapsDispatchTelemetry",
    "InferenceBenchmark",
    "RAW_SCENE_BYTES",
    "RAW_SCENE_MB",
    "SatelliteGnssTelemetry",
    "TrackABridgeResult",
    "TrackBEdgeAlert",
    "TrackBMergedTelemetry",
    "ingest_track_b_into_track_a",
    "run_aerial_verification",
    "run_edge_satellite_inference",
    "track_b_alert_to_event_request",
    "using_pydantic_v2",
]
