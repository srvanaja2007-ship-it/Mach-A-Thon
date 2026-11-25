"""Strict 19-field shared data contract between Track B (Orbit/Edge) and Track A (ClaimPulse)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import re
from typing import Any, Literal, get_args

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

    _PYDANTIC_V2_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when pydantic is absent
    BaseModel = object  # type: ignore[misc, assignment]
    _PYDANTIC_V2_AVAILABLE = False


OrbitPassType = Literal["ascending", "descending"]
DetectedPeril = Literal["flood", "wildfire", "cyclone"]
ResidentialDensityAlert = Literal["negligible", "moderate", "high", "critical"]
ParametricTriggerStatus = Literal["none", "amber_alert", "trigger_met"]
RecommendedActionCode = Literal["auto_payout", "manual_review", "emergency_dispatch"]
DownlinkPriority = Literal["P0_CRITICAL", "P1_HIGH", "P2_ROUTINE"]

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SEVERITY_KEYS = ("low", "medium", "high", "extreme")

EDGE_ALERT_FIELDS: tuple[str, ...] = (
    "satellite_id",
    "orbit_pass_type",
    "scene_timestamp_utc",
    "bounding_box",
    "detected_peril",
    "hazard_polygon_vector",
    "affected_area_km2",
    "peak_hazard_intensity",
    "severity_distribution_pct",
    "inference_confidence",
    "cloud_obstruction_pct",
    "model_version_hash",
    "critical_assets_intercepted",
    "transport_corridors_severed",
    "residential_density_alert",
    "parametric_trigger_status",
    "recommended_action_code",
    "insight_payload_bytes",
    "downlink_priority",
)


def using_pydantic_v2() -> bool:
    """Return True when Pydantic v2 runtime validation is active."""
    return _PYDANTIC_V2_AVAILABLE


def _validate_iso_utc(ts: str) -> str:
    if not isinstance(ts, str) or not _ISO_UTC_RE.match(ts):
        raise ValueError("scene_timestamp_utc must match ISO 8601 UTC format YYYY-MM-DDTHH:MM:SSZ")
    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    return ts


def _validate_bbox(bbox: list[float]) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bounding_box must be a list of 4 floats: [min_lon, min_lat, max_lon, max_lat]")
    vals = [float(x) for x in bbox]
    min_lon, min_lat, max_lon, max_lat = vals
    if not (-180.0 <= min_lon < max_lon <= 180.0):
        raise ValueError("bounding_box requires -180.0 <= min_lon < max_lon <= 180.0")
    if not (-90.0 <= min_lat < max_lat <= 90.0):
        raise ValueError("bounding_box requires -90.0 <= min_lat < max_lat <= 90.0")
    return vals


def _validate_hazard_vector(vec: str) -> str:
    if not isinstance(vec, str) or not vec.strip():
        raise ValueError("hazard_polygon_vector must be a non-empty GeoJSON or WKT string")
    stripped = vec.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict) or "type" not in parsed:
            raise ValueError("hazard_polygon_vector GeoJSON must be an object with 'type'")
    elif not any(stripped.upper().startswith(prefix) for prefix in ("POLYGON", "MULTIPOLYGON")):
        raise ValueError("hazard_polygon_vector must be valid GeoJSON or WKT POLYGON/MULTIPOLYGON")
    return stripped


def _validate_severity_dist(dist: dict[str, float]) -> dict[str, float]:
    if not isinstance(dist, dict):
        raise ValueError("severity_distribution_pct must be a dict")
    missing = [k for k in _SEVERITY_KEYS if k not in dist]
    extra = [k for k in dist if k not in _SEVERITY_KEYS]
    if missing or extra:
        raise ValueError(f"severity_distribution_pct must contain exact keys {_SEVERITY_KEYS}")
    cleaned: dict[str, float] = {}
    for k in _SEVERITY_KEYS:
        val = float(dist[k])
        if val < 0.0 or val > 100.0:
            raise ValueError(f"severity_distribution_pct['{k}'] must be within [0.0, 100.0]")
        cleaned[k] = round(val, 4)
    total = sum(cleaned.values())
    if abs(total - 100.0) > 0.1:
        raise ValueError(f"severity_distribution_pct must sum to 100.0 +/- 0.1 (got {total:.4f})")
    return cleaned


def _compute_wire_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _compute_fixed_point_bytes(payload: dict[str, Any]) -> int:
    work = dict(payload)
    work["insight_payload_bytes"] = 0
    for _ in range(5):
        size = len(_compute_wire_json(work).encode("utf-8"))
        if work["insight_payload_bytes"] == size:
            return size
        work["insight_payload_bytes"] = size
    return int(work["insight_payload_bytes"])


if _PYDANTIC_V2_AVAILABLE:

    class TrackBEdgeAlert(BaseModel):
        """Strict 19-field edge telemetry alert emitted by the satellite onboard AI pipeline."""

        model_config = ConfigDict(extra="forbid", validate_assignment=True)

        # Observation & Orbit
        satellite_id: str = Field(..., min_length=1)
        orbit_pass_type: OrbitPassType
        scene_timestamp_utc: str
        bounding_box: list[float]

        # Hazard Inference (Edge)
        detected_peril: DetectedPeril
        hazard_polygon_vector: str
        affected_area_km2: float = Field(..., ge=0.0)
        peak_hazard_intensity: float = Field(..., ge=0.0)
        severity_distribution_pct: dict[str, float]

        # Quality & Assurance
        inference_confidence: float = Field(..., ge=0.0, le=1.0)
        cloud_obstruction_pct: float = Field(..., ge=0.0, le=100.0)
        model_version_hash: str = Field(..., min_length=1)

        # Edge Exposure Screening
        critical_assets_intercepted: int = Field(..., ge=0)
        transport_corridors_severed: int = Field(..., ge=0)
        residential_density_alert: ResidentialDensityAlert

        # Economic Triggers
        parametric_trigger_status: ParametricTriggerStatus
        recommended_action_code: RecommendedActionCode

        # Downlink Metrics
        insight_payload_bytes: int = Field(default=0, ge=0)
        downlink_priority: DownlinkPriority

        @field_validator("scene_timestamp_utc")
        @classmethod
        def _check_timestamp(cls, v: str) -> str:
            return _validate_iso_utc(v)

        @field_validator("bounding_box")
        @classmethod
        def _check_bbox(cls, v: list[float]) -> list[float]:
            return _validate_bbox(v)

        @field_validator("hazard_polygon_vector")
        @classmethod
        def _check_vector(cls, v: str) -> str:
            return _validate_hazard_vector(v)

        @field_validator("severity_distribution_pct")
        @classmethod
        def _check_severity(cls, v: dict[str, float]) -> dict[str, float]:
            return _validate_severity_dist(v)

        @model_validator(mode="after")
        def _sync_payload_bytes(self) -> "TrackBEdgeAlert":
            # Measure UTF-8 serialized JSON byte length dynamically
            raw = self.model_dump()
            computed = _compute_fixed_point_bytes(raw)
            object.__setattr__(self, "insight_payload_bytes", computed)
            return self

        def to_wire_json(self) -> str:
            """Return canonical compact UTF-8 JSON representation whose byte length equals insight_payload_bytes."""
            return _compute_wire_json(self.model_dump())

        def with_updates(self, **updates: Any) -> "TrackBEdgeAlert":
            """Return a validated copy with updated fields and refreshed insight_payload_bytes."""
            data = self.model_dump()
            data.update(updates)
            data["insight_payload_bytes"] = 0
            return TrackBEdgeAlert(**data)

    class InferenceBenchmark(BaseModel):
        """Onboard compute latency and satellite downlink bandwidth compression metrics."""

        model_config = ConfigDict(extra="forbid")

        raw_scene_mb: float = Field(..., gt=0.0)
        raw_scene_bytes: int = Field(..., gt=0)
        insight_payload_bytes: int = Field(..., gt=0)
        bandwidth_reduction_pct: float = Field(..., ge=0.0, le=100.0)
        inference_latency_ms: float = Field(..., ge=0.0)
        quantization_mode: str = Field(default="INT8")
        onnx_runtime_used: bool = Field(default=False)

    class TrackBMergedTelemetry(BaseModel):
        """Complete validated Track B -> Track A merged operational payload."""

        model_config = ConfigDict(extra="allow")

        scenario_name: str
        edge_alert: TrackBEdgeAlert
        benchmark: InferenceBenchmark
        aerial_verification: dict[str, Any]
        track_a_integration: dict[str, Any]

else:  # pragma: no cover - pure-Python dataclass fallback when pydantic is unavailable

    @dataclass
    class TrackBEdgeAlert:  # type: ignore[no-redef]
        """Dataclass fallback enforcing the exact 19-field Track B contract."""

        satellite_id: str
        orbit_pass_type: OrbitPassType
        scene_timestamp_utc: str
        bounding_box: list[float]
        detected_peril: DetectedPeril
        hazard_polygon_vector: str
        affected_area_km2: float
        peak_hazard_intensity: float
        severity_distribution_pct: dict[str, float]
        inference_confidence: float
        cloud_obstruction_pct: float
        model_version_hash: str
        critical_assets_intercepted: int
        transport_corridors_severed: int
        residential_density_alert: ResidentialDensityAlert
        parametric_trigger_status: ParametricTriggerStatus
        recommended_action_code: RecommendedActionCode
        downlink_priority: DownlinkPriority
        insight_payload_bytes: int = 0

        def __post_init__(self) -> None:
            if not isinstance(self.satellite_id, str) or not self.satellite_id.strip():
                raise ValueError("satellite_id must be a non-empty string")
            if self.orbit_pass_type not in get_args(OrbitPassType):
                raise ValueError("Invalid orbit_pass_type")
            self.scene_timestamp_utc = _validate_iso_utc(self.scene_timestamp_utc)
            self.bounding_box = _validate_bbox(self.bounding_box)
            if self.detected_peril not in get_args(DetectedPeril):
                raise ValueError("Invalid detected_peril")
            self.hazard_polygon_vector = _validate_hazard_vector(self.hazard_polygon_vector)
            if float(self.affected_area_km2) < 0.0:
                raise ValueError("affected_area_km2 must be >= 0.0")
            if float(self.peak_hazard_intensity) < 0.0:
                raise ValueError("peak_hazard_intensity must be >= 0.0")
            self.severity_distribution_pct = _validate_severity_dist(self.severity_distribution_pct)
            if not (0.0 <= float(self.inference_confidence) <= 1.0):
                raise ValueError("inference_confidence must be in [0.0, 1.0]")
            if not (0.0 <= float(self.cloud_obstruction_pct) <= 100.0):
                raise ValueError("cloud_obstruction_pct must be in [0.0, 100.0]")
            if not isinstance(self.model_version_hash, str) or not self.model_version_hash.strip():
                raise ValueError("model_version_hash must be non-empty")
            if int(self.critical_assets_intercepted) < 0:
                raise ValueError("critical_assets_intercepted must be >= 0")
            if int(self.transport_corridors_severed) < 0:
                raise ValueError("transport_corridors_severed must be >= 0")
            if self.residential_density_alert not in get_args(ResidentialDensityAlert):
                raise ValueError("Invalid residential_density_alert")
            if self.parametric_trigger_status not in get_args(ParametricTriggerStatus):
                raise ValueError("Invalid parametric_trigger_status")
            if self.recommended_action_code not in get_args(RecommendedActionCode):
                raise ValueError("Invalid recommended_action_code")
            if self.downlink_priority not in get_args(DownlinkPriority):
                raise ValueError("Invalid downlink_priority")
            self.insight_payload_bytes = _compute_fixed_point_bytes(asdict(self))

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

        def to_wire_json(self) -> str:
            return _compute_wire_json(asdict(self))

        def with_updates(self, **updates: Any) -> "TrackBEdgeAlert":
            data = asdict(self)
            data.update(updates)
            data["insight_payload_bytes"] = 0
            return TrackBEdgeAlert(**data)

    @dataclass
    class InferenceBenchmark:  # type: ignore[no-redef]
        raw_scene_mb: float
        raw_scene_bytes: int
        insight_payload_bytes: int
        bandwidth_reduction_pct: float
        inference_latency_ms: float
        quantization_mode: str = "INT8"
        onnx_runtime_used: bool = False

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass
    class TrackBMergedTelemetry:  # type: ignore[no-redef]
        scenario_name: str
        edge_alert: TrackBEdgeAlert
        benchmark: InferenceBenchmark
        aerial_verification: dict[str, Any] = field(default_factory=dict)
        track_a_integration: dict[str, Any] = field(default_factory=dict)

        def model_dump(self) -> dict[str, Any]:
            return {
                "scenario_name": self.scenario_name,
                "edge_alert": self.edge_alert.model_dump(),
                "benchmark": self.benchmark.model_dump(),
                "aerial_verification": self.aerial_verification,
                "track_a_integration": self.track_a_integration,
            }
