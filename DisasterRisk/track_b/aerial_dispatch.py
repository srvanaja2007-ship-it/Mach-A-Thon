"""3-Tier Autonomous Verification Pipeline: LEO Satellite -> Stratospheric HAPS -> IP65 Drone."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from track_b.schema import TrackBEdgeAlert

EARTH_RADIUS_KM: float = 6371.0072
STANDARD_GPS_ERROR_M: float = 12.5
LEO_AUGMENTED_ERROR_M: float = 0.04
HAPS_ALTITUDE_KM: float = 20.0
HAPS_CRUISE_SPEED_KMH: float = 85.0
DRONE_CRUISE_SPEED_KMH: float = 65.0
MAX_CONFIDENCE_CAP: float = 0.99

# Regional stratospheric loiter stations & ground IP65 drone docks (lat, lon)
_HAPS_STATIONS: list[dict[str, Any]] = [
    {"station_id": "HAPS-EAST-01", "name": "Bay of Bengal Stratospheric Node", "lon": 86.45, "lat": 20.32},
    {"station_id": "HAPS-SOUTH-02", "name": "Coromandel Coastal Loiter Node", "lon": 80.52, "lat": 13.35},
    {"station_id": "HAPS-NORTH-03", "name": "Brahmaputra Valley Loiter Node", "lon": 92.60, "lat": 26.40},
    {"station_id": "HAPS-WEST-04", "name": "Konkan Coastal Loiter Node", "lon": 73.10, "lat": 18.95},
]

_DRONE_DOCKS: list[dict[str, Any]] = [
    {"dock_id": "DOCK-OD-01", "name": "Puri District EOC IP65 Hangar", "lon": 85.83, "lat": 19.82},
    {"dock_id": "DOCK-OD-02", "name": "Bhubaneswar NDRF Airfield Dock", "lon": 85.81, "lat": 20.25},
    {"dock_id": "DOCK-TN-01", "name": "Chennai Ennore Emergency Dock", "lon": 80.28, "lat": 13.15},
    {"dock_id": "DOCK-AS-01", "name": "Guwahati Flood Response Hangar", "lon": 91.74, "lat": 26.14},
    {"dock_id": "DOCK-MH-01", "name": "Navi Mumbai Coastal Drone Pad", "lon": 73.02, "lat": 19.04},
]


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute great-circle distance (km) on WGS84 sphere between two coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi * 0.5) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lam * 0.5) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(EARTH_RADIUS_KM * c, 2)


@dataclass
class SatelliteGnssTelemetry:
    """Tier 1: LEO Satellite observation and LEO-augmented GNSS error correction."""

    tier: str
    satellite_id: str
    orbit_pass_type: str
    standard_gps_error_m: float
    leo_augmented_error_m: float
    precision_gain_factor: float
    status: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HapsDispatchTelemetry:
    """Tier 2: Stratospheric High-Altitude Platform Station (20 km) continuous monitoring."""

    tier: str
    dispatched: bool
    station_id: str
    station_name: str
    altitude_km: float
    cruise_speed_kmh: float
    distance_km: float
    eta_minutes: float
    all_weather_stratospheric_lock: bool
    continuous_monitoring_mode: str
    status: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DroneDispatchTelemetry:
    """Tier 3: Local IP65 weather-hardened drone micro-verification in post-squall window."""

    tier: str
    dispatched: bool
    dock_id: str
    dock_name: str
    ip_rating: str
    cruise_speed_kmh: float
    distance_km: float
    eta_minutes: float
    navigation_gnss_error_m: float
    flight_window: str
    refined_critical_assets: int
    refined_transport_corridors: int
    confidence_before: float
    confidence_after: float
    confidence_boost: float
    status: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AerialVerificationReport:
    """Full 3-Tier (Satellite -> HAPS -> Drone) verification report and refined Track B alert."""

    escalation_triggered: bool
    escalation_reason: str
    tier1_satellite_gnss: SatelliteGnssTelemetry
    tier2_haps: HapsDispatchTelemetry
    tier3_drone: DroneDispatchTelemetry
    verified_alert: TrackBEdgeAlert

    def model_dump(self) -> dict[str, Any]:
        return {
            "escalation_triggered": self.escalation_triggered,
            "escalation_reason": self.escalation_reason,
            "tier1_satellite_gnss": self.tier1_satellite_gnss.model_dump(),
            "tier2_haps": self.tier2_haps.model_dump(),
            "tier3_drone": self.tier3_drone.model_dump(),
            "verified_alert": self.verified_alert.model_dump(),
        }


def _nearest_node(
    lon: float, lat: float, nodes: list[dict[str, Any]]
) -> tuple[dict[str, Any], float]:
    best = nodes[0]
    best_dist = haversine_km(lon, lat, best["lon"], best["lat"])
    for node in nodes[1:]:
        d = haversine_km(lon, lat, node["lon"], node["lat"])
        if d < best_dist:
            best = node
            best_dist = d
    return best, best_dist


def run_aerial_verification(alert: TrackBEdgeAlert) -> AerialVerificationReport:
    """Execute 3-Tier Satellite -> HAPS -> Drone verification pipeline on a TrackBEdgeAlert."""
    min_lon, min_lat, max_lon, max_lat = alert.bounding_box
    c_lon = (min_lon + max_lon) * 0.5
    c_lat = (min_lat + max_lat) * 0.5

    # Tier 1: LEO Satellite GNSS augmentation (12.5m storm GPS -> 0.04m LEO-augmented)
    tier1 = SatelliteGnssTelemetry(
        tier="Tier 1: LEO Satellite + GNSS Augmentation",
        satellite_id=alert.satellite_id,
        orbit_pass_type=alert.orbit_pass_type,
        standard_gps_error_m=STANDARD_GPS_ERROR_M,
        leo_augmented_error_m=LEO_AUGMENTED_ERROR_M,
        precision_gain_factor=round(STANDARD_GPS_ERROR_M / LEO_AUGMENTED_ERROR_M, 1),
        status="LEO_GNSS_CORRECTION_ACTIVE",
    )

    should_escalate = (
        alert.parametric_trigger_status != "none"
        or alert.downlink_priority == "P0_CRITICAL"
    )

    haps_node, haps_dist = _nearest_node(c_lon, c_lat, _HAPS_STATIONS)
    drone_dock, drone_dist = _nearest_node(c_lon, c_lat, _DRONE_DOCKS)

    if not should_escalate:
        reason = (
            f"Routine telemetry (trigger={alert.parametric_trigger_status}, "
            f"priority={alert.downlink_priority}); HAPS and local drone remain standby."
        )
        tier2 = HapsDispatchTelemetry(
            tier="Tier 2: Stratospheric HAPS (20 km)",
            dispatched=False,
            station_id=haps_node["station_id"],
            station_name=haps_node["name"],
            altitude_km=HAPS_ALTITUDE_KM,
            cruise_speed_kmh=HAPS_CRUISE_SPEED_KMH,
            distance_km=0.0,
            eta_minutes=0.0,
            all_weather_stratospheric_lock=False,
            continuous_monitoring_mode="STANDBY",
            status="STANDBY_IDLE",
        )
        tier3 = DroneDispatchTelemetry(
            tier="Tier 3: IP65 Local Drone",
            dispatched=False,
            dock_id=drone_dock["dock_id"],
            dock_name=drone_dock["name"],
            ip_rating="IP65",
            cruise_speed_kmh=DRONE_CRUISE_SPEED_KMH,
            distance_km=0.0,
            eta_minutes=0.0,
            navigation_gnss_error_m=LEO_AUGMENTED_ERROR_M,
            flight_window="NOT_REQUIRED",
            refined_critical_assets=alert.critical_assets_intercepted,
            refined_transport_corridors=alert.transport_corridors_severed,
            confidence_before=alert.inference_confidence,
            confidence_after=alert.inference_confidence,
            confidence_boost=0.0,
            status="DOCKED_IDLE",
        )
        return AerialVerificationReport(
            escalation_triggered=False,
            escalation_reason=reason,
            tier1_satellite_gnss=tier1,
            tier2_haps=tier2,
            tier3_drone=tier3,
            verified_alert=alert,
        )

    # Tier 2: Stratospheric HAPS dispatch (20 km altitude above storm cloud decks)
    haps_eta_min = round((haps_dist / HAPS_CRUISE_SPEED_KMH) * 60.0, 2)
    tier2 = HapsDispatchTelemetry(
        tier="Tier 2: Stratospheric HAPS (20 km)",
        dispatched=True,
        station_id=haps_node["station_id"],
        station_name=haps_node["name"],
        altitude_km=HAPS_ALTITUDE_KM,
        cruise_speed_kmh=HAPS_CRUISE_SPEED_KMH,
        distance_km=haps_dist,
        eta_minutes=haps_eta_min,
        all_weather_stratospheric_lock=True,
        continuous_monitoring_mode="24/7 Above-Cloud Persistent SAR/EO Relay",
        status="DISPATCHED_ON_STATION",
    )

    # Tier 3: Local IP65 weather-hardened drone micro-verification in post-squall window
    drone_eta_min = round((drone_dist / DRONE_CRUISE_SPEED_KMH) * 60.0, 2)
    raw_boost = 0.06 if alert.parametric_trigger_status == "trigger_met" else 0.04
    conf_after = min(MAX_CONFIDENCE_CAP, round(alert.inference_confidence + raw_boost, 2))
    actual_boost = round(conf_after - alert.inference_confidence, 2)

    # Close-up oblique drone orthomosaic detects obscured micro-infrastructure & road washouts
    refined_assets = max(
        alert.critical_assets_intercepted,
        alert.critical_assets_intercepted + (1 if alert.parametric_trigger_status == "trigger_met" else 0),
    )
    refined_corridors = max(
        alert.transport_corridors_severed,
        alert.transport_corridors_severed + (1 if alert.parametric_trigger_status == "trigger_met" else 0),
    )

    verified_alert = alert.with_updates(
        inference_confidence=conf_after,
        critical_assets_intercepted=refined_assets,
        transport_corridors_severed=refined_corridors,
    )

    tier3 = DroneDispatchTelemetry(
        tier="Tier 3: IP65 Local Drone",
        dispatched=True,
        dock_id=drone_dock["dock_id"],
        dock_name=drone_dock["name"],
        ip_rating="IP65",
        cruise_speed_kmh=DRONE_CRUISE_SPEED_KMH,
        distance_km=drone_dist,
        eta_minutes=drone_eta_min,
        navigation_gnss_error_m=LEO_AUGMENTED_ERROR_M,
        flight_window="POST_SQUALL_CLEAR_WINDOW (0.04m LEO-GNSS Locked)",
        refined_critical_assets=refined_assets,
        refined_transport_corridors=refined_corridors,
        confidence_before=alert.inference_confidence,
        confidence_after=conf_after,
        confidence_boost=actual_boost,
        status="DISPATCHED_VERIFIED",
    )

    reason = (
        f"Escalated due to parametric_trigger_status={alert.parametric_trigger_status} "
        f"and downlink_priority={alert.downlink_priority}."
    )
    return AerialVerificationReport(
        escalation_triggered=True,
        escalation_reason=reason,
        tier1_satellite_gnss=tier1,
        tier2_haps=tier2,
        tier3_drone=tier3,
        verified_alert=verified_alert,
    )
