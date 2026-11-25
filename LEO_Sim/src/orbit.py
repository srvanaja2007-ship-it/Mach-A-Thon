"""
Orbital mechanics for a circular LEO satellite.

Orbital parameters (fixed, per project spec):
  - Altitude    : 500 km
  - Inclination : 97.4 deg  (Sun-synchronous)
  - Shape       : Circular

Derived (computed, not hardcoded):
  - Semi-major axis: a = R_earth + altitude
  - Orbital period: T = 2π √(a³/μ)   [Kepler 3rd law]
  - Angular velocity: ω = 2π/T

Ground station: Tumkur, Karnataka, India (13.3379° N, 77.1173° E)

NOTE: orbit simulation is SIMULATED for visualization purposes.
      Real VIIRS data processing is separate (eco_act_adapter).
"""

import math
import numpy as np
from dataclasses import dataclass

# ── Physical constants ────────────────────────────────────────────────────────
MU_EARTH = 3.986004418e5   # km³ s⁻²  (Earth's gravitational parameter)
R_EARTH  = 6371.0          # km       (mean Earth radius)

# ── Default orbital parameters ────────────────────────────────────────────────
DEFAULT_ALTITUDE_KM   = 500.0    # km
DEFAULT_INCLINATION_DEG = 97.4   # degrees (Sun-synchronous LEO)

# ── Ground station ─────────────────────────────────────────────────────────────
GROUND_STATION = {
    "name": "Tumkur Ground Station",
    "city": "Tumkur, Karnataka, India",
    "lat":  13.3379,
    "lon":  77.1173,
}


@dataclass
class OrbitalParams:
    altitude_km: float = DEFAULT_ALTITUDE_KM
    inclination_deg: float = DEFAULT_INCLINATION_DEG

    @property
    def semi_major_axis_km(self) -> float:
        return R_EARTH + self.altitude_km

    @property
    def period_seconds(self) -> float:
        """Orbital period from Kepler's 3rd law: T = 2π √(a³/μ)"""
        a = self.semi_major_axis_km
        return 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH)

    @property
    def period_minutes(self) -> float:
        return self.period_seconds / 60.0

    @property
    def angular_velocity_deg_per_sec(self) -> float:
        """ω in degrees per second"""
        return 360.0 / self.period_seconds

    @property
    def velocity_km_per_s(self) -> float:
        """Circular orbital velocity: v = √(μ/a)"""
        return math.sqrt(MU_EARTH / self.semi_major_axis_km)


class Satellite:
    """
    Simulated LEO satellite with circular Keplerian orbit.
    Position returned as (lat, lon) in degrees for Plotly globe display.
    SIMULATED: orbital motion is computed analytically for visualization only.
    """

    def __init__(self, params: OrbitalParams | None = None):
        self.params = params or OrbitalParams()
        # Initial RAAN set so that the ascending node passes near Tumkur longitude (77°E)
        # at t=0: lon = RAAN - gst(0) = RAAN. Set to 77 so orbit starts over India.
        self._raan = 77.0          # degrees – initial Right Ascension of Ascending Node
        self._earth_rotation_rate = 360.0 / 86164.0  # deg/s (sidereal day)

    # ── Position calculation ──────────────────────────────────────────────────

    def position_at(self, elapsed_seconds: float) -> dict:
        """
        Return satellite lat/lon/alt at *elapsed_seconds* from epoch.

        Uses simplified (but physically motivated) calculation:
          - true anomaly M = ω * t  (circular orbit, M = ν)
          - inclination I
          - Earth rotation shifts sub-satellite longitude westward
        """
        inc = math.radians(self.params.inclination_deg)
        T   = self.params.period_seconds
        omega = 2.0 * math.pi / T

        # Argument of latitude (circular orbit: M = ν)
        u = omega * elapsed_seconds          # radians

        # Position in orbital plane
        x_orb = math.cos(u)
        y_orb = math.sin(u)
        z_orb = 0.0

        # Rotate by inclination (rotation around X-axis)
        x1 = x_orb
        y1 = y_orb * math.cos(inc)
        z1 = y_orb * math.sin(inc)

        # Earth has rotated; compute RAAN drift (Sun-synchronous: ~0.9856 deg/day)
        raan_drift_rate = math.radians(0.9856 / 86400.0)  # rad/s
        raan = math.radians(self._raan) + raan_drift_rate * elapsed_seconds

        # Rotate by RAAN (rotation around Z-axis)
        x2 = x1 * math.cos(raan) - y1 * math.sin(raan)
        y2 = x1 * math.sin(raan) + y1 * math.cos(raan)
        z2 = z1

        # Earth-Fixed frame: subtract Earth rotation
        gst = math.radians(self._earth_rotation_rate * elapsed_seconds)
        x_ef = x2 * math.cos(gst) + y2 * math.sin(gst)
        y_ef = -x2 * math.sin(gst) + y2 * math.cos(gst)
        z_ef = z2

        # Geodetic lat/lon (spherical approximation)
        lat = math.degrees(math.asin(max(-1.0, min(1.0, z_ef))))
        raw_lon = math.degrees(math.atan2(y_ef, x_ef))
        # Normalise to [-180, 180]: use ((raw+180)%360)-180
        # (the simpler raw%360-180 gives -180 for raw=0, which is wrong)
        lon = ((raw_lon + 180.0) % 360.0) - 180.0

        return {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "altitude_km": self.params.altitude_km,
            "elapsed_s": elapsed_seconds,
            "orbit_fraction": (elapsed_seconds % T) / T,
            "is_simulated": True,
        }

    def position_eci_km(self, elapsed_seconds: float) -> np.ndarray:
        """Return the propagated ECI position in kilometres.

        This is a two-body, circular-orbit model.  RAAN drift is included as a
        slow visual approximation; perturbations, drag and attitude are not.
        """
        inc = math.radians(self.params.inclination_deg)
        u = (2.0 * math.pi / self.params.period_seconds) * elapsed_seconds
        r = self.params.semi_major_axis_km
        x_orb, y_orb = r * math.cos(u), r * math.sin(u)
        raan = math.radians(self._raan) + math.radians(0.9856) * elapsed_seconds / 86400.0
        x1, y1, z1 = x_orb, y_orb * math.cos(inc), y_orb * math.sin(inc)
        return np.array([
            x1 * math.cos(raan) - y1 * math.sin(raan),
            x1 * math.sin(raan) + y1 * math.cos(raan),
            z1,
        ], dtype=float)

    def position_ecef_km(self, elapsed_seconds: float) -> np.ndarray:
        """Convert the propagated ECI vector to ECEF using sidereal rotation."""
        theta = math.radians(self._earth_rotation_rate * elapsed_seconds)
        x, y, z = self.position_eci_km(elapsed_seconds)
        return np.array([
            x * math.cos(theta) + y * math.sin(theta),
            -x * math.sin(theta) + y * math.cos(theta),
            z,
        ], dtype=float)

    @staticmethod
    def _ground_station_ecef_km(gs_lat: float, gs_lon: float) -> np.ndarray:
        lat, lon = math.radians(gs_lat), math.radians(gs_lon)
        return R_EARTH * np.array([
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ], dtype=float)

    def look_angles(self, elapsed_seconds: float, gs_lat: float, gs_lon: float) -> dict:
        """Return slant range, elevation and azimuth from a ground station."""
        sat = self.position_ecef_km(elapsed_seconds)
        gs = self._ground_station_ecef_km(gs_lat, gs_lon)
        delta = sat - gs
        distance = float(np.linalg.norm(delta))
        lat, lon = math.radians(gs_lat), math.radians(gs_lon)
        east = -math.sin(lon) * delta[0] + math.cos(lon) * delta[1]
        north = (-math.sin(lat) * math.cos(lon) * delta[0]
                 - math.sin(lat) * math.sin(lon) * delta[1]
                 + math.cos(lat) * delta[2])
        up = (math.cos(lat) * math.cos(lon) * delta[0]
              + math.cos(lat) * math.sin(lon) * delta[1]
              + math.sin(lat) * delta[2])
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, up / distance))))
        azimuth = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
        return {
            "distance_km": round(distance, 3),
            "elevation_deg": round(elevation, 3),
            "azimuth_deg": round(azimuth, 3),
        }

    def visibility_at(self, elapsed_seconds: float, gs_lat: float, gs_lon: float,
                      min_elevation_deg: float = 5.0) -> dict:
        angles = self.look_angles(elapsed_seconds, gs_lat, gs_lon)
        return {**angles, "visible": angles["elevation_deg"] >= min_elevation_deg,
                "min_elevation_deg": min_elevation_deg}

    def next_visibility_seconds(self, elapsed_seconds: float, gs_lat: float, gs_lon: float,
                                min_elevation_deg: float = 5.0,
                                step_seconds: float = 10.0) -> float | None:
        """Find the next visibility start within four orbital periods by sampling."""
        start = self.visibility_at(elapsed_seconds, gs_lat, gs_lon, min_elevation_deg)["visible"]
        for offset in np.arange(step_seconds, 4.0 * self.params.period_seconds + step_seconds, step_seconds):
            current = self.visibility_at(elapsed_seconds + float(offset), gs_lat, gs_lon,
                                         min_elevation_deg)["visible"]
            if current and not start:
                return round(float(offset), 1)
            start = current
        return None

    # ── Orbit ground-track ────────────────────────────────────────────────────

    def ground_track(self, n_points: int = 360) -> list[dict]:
        """Return one full orbit ground-track as a list of {lat, lon} dicts."""
        T = self.params.period_seconds
        step = T / n_points
        return [self.position_at(i * step) for i in range(n_points + 1)]

    # ── Communication window ──────────────────────────────────────────────────

    def is_in_view(self, sat_lat: float, sat_lon: float,
                   gs_lat: float, gs_lon: float,
                   min_elevation_deg: float = 5.0) -> bool:
        """
        Check if satellite is above min_elevation above the ground station.
        Uses spherical Earth geometry (simulated).
        """
        # Backward-compatible helper for callers that only have lat/lon.
        # Use the same spherical geometry as visibility_at.
        sat_lat_r = math.radians(sat_lat)
        sat_lon_r = math.radians(sat_lon)
        gs_lat_r  = math.radians(gs_lat)
        gs_lon_r  = math.radians(gs_lon)

        d_lon = sat_lon_r - gs_lon_r
        cos_c = (math.sin(gs_lat_r) * math.sin(sat_lat_r) +
                 math.cos(gs_lat_r) * math.cos(sat_lat_r) * math.cos(d_lon))
        cos_c = max(-1.0, min(1.0, cos_c))
        rho = math.acos(cos_c)                    # central angle (rad)

        a = self.params.altitude_km
        rho_deg = math.degrees(rho)

        # Elevation angle from ground station (spherical geometry)
        if rho_deg >= 90.0:
            return False
        num   = math.cos(rho) - R_EARTH / (R_EARTH + a)
        denom = math.sin(rho)
        if abs(denom) < 1e-9:
            return False
        el = math.degrees(math.atan(num / denom))
        return el >= min_elevation_deg

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        p = self.params
        return {
            "altitude_km":           p.altitude_km,
            "inclination_deg":       p.inclination_deg,
            "semi_major_axis_km":    round(p.semi_major_axis_km, 2),
            "orbital_period_seconds": round(p.period_seconds, 2),
            "orbital_period_minutes": round(p.period_minutes, 3),
            "velocity_km_per_s":     round(p.velocity_km_per_s, 4),
            "note": "[SIMULATED] Orbit visualization only",
        }
