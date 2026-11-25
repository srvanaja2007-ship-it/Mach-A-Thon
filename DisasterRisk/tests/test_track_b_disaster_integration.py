"""Comprehensive pytest integration suite for Track B (Orbit Edge AI) -> Track A (ClaimPulse)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

_DISASTER_RISK_ROOT = Path(__file__).resolve().parent.parent
if str(_DISASTER_RISK_ROOT) not in sys.path:
    sys.path.insert(0, str(_DISASTER_RISK_ROOT))

from backend.core.schemas import EventRequest  # noqa: E402
from backend.exporters.export_service import export_csv, export_geojson, export_pdf  # noqa: E402
from backend.main import EVENT_STORE, app  # noqa: E402
from backend.modules.implementations import INR_SCALE  # noqa: E402
from backend.services.analyst import answer_question  # noqa: E402
from run_merged_disaster_pipeline import run_demo_scenarios  # noqa: E402
from track_b.aerial_dispatch import (  # noqa: E402
    DRONE_CRUISE_SPEED_KMH,
    HAPS_ALTITUDE_KM,
    HAPS_CRUISE_SPEED_KMH,
    LEO_AUGMENTED_ERROR_M,
    MAX_CONFIDENCE_CAP,
    STANDARD_GPS_ERROR_M,
    run_aerial_verification,
)
from track_b.bridge import ingest_track_b_into_track_a, track_b_alert_to_event_request  # noqa: E402
from track_b.edge_inference import RAW_SCENE_BYTES, RAW_SCENE_MB, run_edge_satellite_inference  # noqa: E402
from track_b.schema import EDGE_ALERT_FIELDS, TrackBEdgeAlert  # noqa: E402


def _asgi_call(method: str, path: str, json_body: dict[str, object] | None = None) -> tuple[int, dict[str, object] | list[object]]:
    """Invoke the Starlette ASGI app directly without external testclient dependencies."""
    raw_body = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    headers = [(b"content-type", b"application/json")] if json_body is not None else []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    sent_body = False
    status_code = 500
    chunks: list[bytes] = []

    async def receive() -> dict[str, object]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])  # type: ignore[arg-type]
        elif message["type"] == "http.response.body":
            chunks.append(bytes(message.get("body", b"")))  # type: ignore[arg-type]

    asyncio.run(app(scope, receive, send))
    return status_code, json.loads(b"".join(chunks).decode("utf-8"))


def _make_valid_alert(**overrides: object) -> TrackBEdgeAlert:
    base: dict[str, object] = {
        "satellite_id": "EOS-04",
        "orbit_pass_type": "descending",
        "scene_timestamp_utc": "2026-10-22T06:14:00Z",
        "bounding_box": [85.72, 19.76, 86.38, 20.24],
        "detected_peril": "flood",
        "hazard_polygon_vector": json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [85.85, 19.85],
                        [86.20, 19.85],
                        [86.20, 20.12],
                        [85.85, 20.12],
                        [85.85, 19.85],
                    ]
                ],
            }
        ),
        "affected_area_km2": 325.4,
        "peak_hazard_intensity": 2.15,
        "severity_distribution_pct": {"low": 15.0, "medium": 25.0, "high": 35.0, "extreme": 25.0},
        "inference_confidence": 0.91,
        "cloud_obstruction_pct": 0.0,
        "model_version_hash": "v2.4-quantized-int8-a1b2c3d4",
        "critical_assets_intercepted": 3,
        "transport_corridors_severed": 2,
        "residential_density_alert": "high",
        "parametric_trigger_status": "trigger_met",
        "recommended_action_code": "auto_payout",
        "downlink_priority": "P0_CRITICAL",
    }
    base.update(overrides)
    return TrackBEdgeAlert(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. 19-Field Schema & Validator Tests
# ---------------------------------------------------------------------------
def test_schema_has_exact_19_fields_and_dynamic_bytes() -> None:
    alert = _make_valid_alert()
    dumped = alert.model_dump()
    assert len(dumped) == 19
    assert set(dumped.keys()) == set(EDGE_ALERT_FIELDS)

    wire_bytes = len(alert.to_wire_json().encode("utf-8"))
    assert alert.insight_payload_bytes == wire_bytes
    assert alert.insight_payload_bytes > 0

    updated = alert.with_updates(inference_confidence=0.97, critical_assets_intercepted=5)
    assert updated.insight_payload_bytes == len(updated.to_wire_json().encode("utf-8"))


@pytest.mark.parametrize(
    "bad_bbox",
    [
        [85.72, 19.76, 86.38],  # wrong length
        [86.38, 19.76, 85.72, 20.24],  # min_lon >= max_lon
        [85.72, 20.24, 86.38, 19.76],  # min_lat >= max_lat
        [-190.0, 19.76, 86.38, 20.24],  # out of range
    ],
)
def test_schema_rejects_invalid_bounding_box(bad_bbox: list[float]) -> None:
    with pytest.raises(Exception):
        _make_valid_alert(bounding_box=bad_bbox)


@pytest.mark.parametrize(
    "bad_dist",
    [
        {"low": 10.0, "medium": 20.0, "high": 30.0, "extreme": 10.0},  # sums to 70
        {"low": 50.0, "medium": 50.0, "high": 10.0, "extreme": 0.0},  # sums to 110
        {"low": 25.0, "medium": 25.0, "high": 50.0},  # missing extreme
        {"low": -5.0, "medium": 35.0, "high": 40.0, "extreme": 30.0},  # negative
    ],
)
def test_schema_rejects_invalid_severity_distribution(bad_dist: dict[str, float]) -> None:
    with pytest.raises(Exception):
        _make_valid_alert(severity_distribution_pct=bad_dist)


def test_schema_rejects_bad_timestamps_literals_and_bounds() -> None:
    with pytest.raises(Exception):
        _make_valid_alert(scene_timestamp_utc="2026-10-22 06:14:00")
    with pytest.raises(Exception):
        _make_valid_alert(orbit_pass_type="polar")
    with pytest.raises(Exception):
        _make_valid_alert(detected_peril="earthquake")
    with pytest.raises(Exception):
        _make_valid_alert(inference_confidence=1.25)
    with pytest.raises(Exception):
        _make_valid_alert(cloud_obstruction_pct=-1.0)
    with pytest.raises(Exception):
        _make_valid_alert(hazard_polygon_vector="NOT_GEO_DATA")


# ---------------------------------------------------------------------------
# 2. Edge Inference & Bandwidth Compression Tests
# ---------------------------------------------------------------------------
def test_edge_inference_bandwidth_savings_and_latency() -> None:
    res = run_edge_satellite_inference(
        satellite_id="EOS-04",
        orbit_pass_type="descending",
        bounding_box=[85.72, 19.76, 86.38, 20.24],
        detected_peril="flood",
        peak_hazard_intensity=2.15,
        cloud_obstruction_pct=0.0,
    )
    alert = res.alert
    bench = res.benchmark

    assert bench.raw_scene_mb == RAW_SCENE_MB
    assert bench.raw_scene_bytes == RAW_SCENE_BYTES
    assert alert.insight_payload_bytes < 3072  # < 3 KB
    assert bench.bandwidth_reduction_pct >= 99.99
    assert bench.inference_latency_ms >= 0.0
    assert alert.model_version_hash.startswith("v2.4-quantized-int8")
    assert alert.affected_area_km2 > 0.0
    assert abs(sum(alert.severity_distribution_pct.values()) - 100.0) <= 0.1

    geojson_obj = json.loads(alert.hazard_polygon_vector)
    assert geojson_obj["type"] == "Polygon"
    assert geojson_obj["coordinates"][0][0] == geojson_obj["coordinates"][0][-1]


@pytest.mark.parametrize(
    ("peril", "intensity", "expected_trigger"),
    [
        ("flood", 2.15, "trigger_met"),
        ("wildfire", 0.78, "trigger_met"),
        ("cyclone", 145.0, "trigger_met"),
        ("flood", 0.40, "none"),
    ],
)
def test_edge_inference_supports_all_perils(
    peril: str, intensity: float, expected_trigger: str
) -> None:
    res = run_edge_satellite_inference(
        satellite_id="EOS-04",
        detected_peril=peril,  # type: ignore[arg-type]
        peak_hazard_intensity=intensity,
        cloud_obstruction_pct=0.0,
    )
    assert res.alert.detected_peril == peril
    assert res.alert.parametric_trigger_status == expected_trigger


# ---------------------------------------------------------------------------
# 3. 3-Tier Aerial Verification Escalation Tests
# ---------------------------------------------------------------------------
def test_aerial_escalation_triggers_on_p0_critical() -> None:
    edge_res = run_edge_satellite_inference(
        satellite_id="EOS-04",
        orbit_pass_type="descending",
        bounding_box=[85.72, 19.76, 86.38, 20.24],
        detected_peril="flood",
        peak_hazard_intensity=2.15,
        cloud_obstruction_pct=0.0,
    )
    assert edge_res.alert.downlink_priority == "P0_CRITICAL"
    assert edge_res.alert.parametric_trigger_status == "trigger_met"
    assert edge_res.alert.recommended_action_code == "auto_payout"

    report = run_aerial_verification(edge_res.alert)
    assert report.escalation_triggered is True

    # Tier 1 LEO GNSS error correction
    t1 = report.tier1_satellite_gnss
    assert t1.standard_gps_error_m == STANDARD_GPS_ERROR_M
    assert t1.leo_augmented_error_m == LEO_AUGMENTED_ERROR_M
    assert t1.precision_gain_factor > 100.0

    # Tier 2 Stratospheric HAPS
    t2 = report.tier2_haps
    assert t2.dispatched is True
    assert t2.altitude_km == HAPS_ALTITUDE_KM
    assert t2.cruise_speed_kmh == HAPS_CRUISE_SPEED_KMH
    assert t2.all_weather_stratospheric_lock is True
    assert t2.distance_km > 0.0
    assert t2.eta_minutes > 0.0

    # Tier 3 IP65 Drone refinement & confidence cap
    t3 = report.tier3_drone
    assert t3.dispatched is True
    assert t3.cruise_speed_kmh == DRONE_CRUISE_SPEED_KMH
    assert t3.navigation_gnss_error_m == LEO_AUGMENTED_ERROR_M
    assert t3.confidence_after > t3.confidence_before
    assert t3.confidence_after <= MAX_CONFIDENCE_CAP
    assert report.verified_alert.critical_assets_intercepted >= edge_res.alert.critical_assets_intercepted
    assert report.verified_alert.transport_corridors_severed >= edge_res.alert.transport_corridors_severed
    assert report.verified_alert.insight_payload_bytes == len(
        report.verified_alert.to_wire_json().encode("utf-8")
    )


def test_aerial_confidence_never_exceeds_0_99_cap() -> None:
    high_conf_alert = _make_valid_alert(inference_confidence=0.97)
    report = run_aerial_verification(high_conf_alert)
    assert report.tier3_drone.dispatched is True
    assert report.verified_alert.inference_confidence == 0.99


def test_aerial_stays_idle_on_p2_routine() -> None:
    edge_res = run_edge_satellite_inference(
        satellite_id="RISAT-1A",
        orbit_pass_type="ascending",
        bounding_box=[85.94, 19.93, 86.08, 20.03],
        detected_peril="flood",
        peak_hazard_intensity=0.40,
        cloud_obstruction_pct=0.0,
    )
    assert edge_res.alert.downlink_priority == "P2_ROUTINE"
    assert edge_res.alert.parametric_trigger_status == "none"
    assert edge_res.alert.recommended_action_code == "manual_review"

    report = run_aerial_verification(edge_res.alert)
    assert report.escalation_triggered is False
    assert report.tier2_haps.dispatched is False
    assert report.tier2_haps.all_weather_stratospheric_lock is False
    assert report.tier2_haps.status == "STANDBY_IDLE"
    assert report.tier3_drone.dispatched is False
    assert report.tier3_drone.confidence_boost == 0.0
    assert report.tier3_drone.status == "DOCKED_IDLE"
    assert report.verified_alert == edge_res.alert


# ---------------------------------------------------------------------------
# 4. End-to-End Track A Integration & Backward Compatibility Tests
# ---------------------------------------------------------------------------
def test_end_to_end_track_a_bridge_and_http_ingest() -> None:
    edge_res = run_edge_satellite_inference(
        satellite_id="EOS-04",
        orbit_pass_type="descending",
        bounding_box=[85.72, 19.76, 86.38, 20.24],
        detected_peril="flood",
        peak_hazard_intensity=2.15,
        cloud_obstruction_pct=0.0,
    )
    aerial = run_aerial_verification(edge_res.alert)
    req = track_b_alert_to_event_request(aerial.verified_alert)
    assert req.disaster_type == "flood"
    assert req.bbox is not None

    bridge_res = ingest_track_b_into_track_a(aerial.verified_alert, aerial_report=aerial)
    assert bridge_res.executed_action_code == "auto_payout"
    assert bridge_res.total_expected_loss_inr > 0.0
    assert bridge_res.total_expected_loss_usd == pytest.approx(
        bridge_res.total_expected_loss_inr / INR_SCALE, abs=0.02
    )
    assert bridge_res.claims_fast_tracked_count > 0
    assert "Track B Edge-Verified Action Executed: auto_payout" in bridge_res.latest_audit_action

    # Verify HTTP API endpoint also ingests Track B payloads cleanly
    status, body = _asgi_call(
        "POST",
        "/api/track-b/ingest",
        {
            "edge_alert": aerial.verified_alert.model_dump(),
            "aerial_verification": aerial.model_dump(),
            "district": "Puri & Kendrapara",
        },
    )
    assert status == 200
    assert isinstance(body, dict)
    assert body["loss"]["total_expected_loss_inr"] > 0.0
    assert body["decisions"]["claims_fast_tracked_count"] > 0


def test_existing_track_a_behavior_backward_compatible() -> None:
    assert "odisha-flood-2026" in EVENT_STORE
    bundle = EVENT_STORE["odisha-flood-2026"]
    assert bundle.loss.total_expected_loss_inr > 0.0
    assert len(export_csv(bundle)) > 100
    assert json.loads(export_geojson(bundle))["type"] == "FeatureCollection"
    assert export_pdf(bundle).startswith(b"%PDF")

    ask = answer_question(bundle, "What are the highest loss zones?", EventRequest(preset_id="odisha-flood-2026"))
    assert len(ask.citations) > 0

    health_status, health_body = _asgi_call("GET", "/api/health")
    assert health_status == 200
    assert isinstance(health_body, dict)
    assert health_body["status"] == "ok"
    events_status, events_body = _asgi_call("GET", "/api/events")
    assert events_status == 200
    assert isinstance(events_body, list)
    assert len(events_body) >= 9


def test_merged_demo_runner_exports_valid_json(tmp_path: Path) -> None:
    out_file = tmp_path / "merged_disaster_alert.json"
    results, written_path = run_demo_scenarios(output_path=out_file)
    assert written_path == out_file
    assert len(results) == 2
    assert results[0].edge_alert.downlink_priority == "P0_CRITICAL"
    assert results[0].edge_alert.recommended_action_code == "auto_payout"
    assert results[0].aerial_verification["escalation_triggered"] is True
    assert results[1].edge_alert.downlink_priority == "P2_ROUTINE"
    assert results[1].edge_alert.recommended_action_code == "manual_review"
    assert results[1].aerial_verification["escalation_triggered"] is False

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(loaded["scenarios"]) == 2
    assert len(loaded["primary_scenario"]["edge_alert"]) == 19
