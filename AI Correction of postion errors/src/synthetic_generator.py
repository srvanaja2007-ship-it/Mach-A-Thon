"""
Synthetic GNSS Trajectory Generator with Realistic Multi-Path & Urban Canyon Noise.
Clearly labeled as [SIMULATED DATA].
"""
import numpy as np
import pandas as pd
from src.config import METERS_PER_DEG_LAT


def generate_synthetic_trajectory(
    num_points: int = 500,
    base_lat: float = 37.4221,  # Mountain View, CA
    base_lon: float = -122.0841,
    dt_seconds: float = 1.0,
    urban_canyon: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic vehicle trajectory with true coordinates and noisy GPS observations.
    
    Noise incorporates:
    1. Atmospheric / ionospheric slow drift (autocorrelated noise).
    2. Dynamic vehicle motion lag (Kalman filter lag during turns and accelerations).
    3. Multi-path reflections (sudden lateral jumps in urban canyon sections).
    
    Returns DataFrame clearly marked as [SIMULATED].
    """
    np.random.seed(seed)
    
    t = np.arange(num_points) * dt_seconds
    millis = (1620000000 + t) * 1000  # Unix timestamp in ms
    
    # Generate smooth 2D vehicle path (S-curves + straight highway + city turns)
    speed_profile = 12.0 + 5.0 * np.sin(0.02 * t)  # 7 to 17 m/s (~25 - 60 km/h)
    
    # Heading changes with time (simulating turns)
    heading_curve = 0.5 * np.sin(0.015 * t) + 0.3 * np.cos(0.04 * t)
    
    # Cumulative trajectory in local metric meters (East, North)
    dx = speed_profile * np.sin(heading_curve) * dt_seconds
    dy = speed_profile * np.cos(heading_curve) * dt_seconds
    
    x_gt = np.cumsum(dx)
    y_gt = np.cumsum(dy)
    
    # Convert metric ground truth to lat / lon
    lat_rad_base = np.radians(base_lat)
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(lat_rad_base)
    
    lat_gt = base_lat + (y_gt / METERS_PER_DEG_LAT)
    lon_gt = base_lon + (x_gt / meters_per_deg_lon)
    
    # Generate realistic GPS error components
    # 1. Autocorrelated multipath / atmospheric drift (AR(1) process)
    drift_e = np.zeros(num_points)
    drift_n = np.zeros(num_points)
    phi = 0.95  # High temporal autocorrelation
    
    noise_e = np.random.normal(0, 0.6, num_points)
    noise_n = np.random.normal(0, 0.6, num_points)
    for i in range(1, num_points):
        drift_e[i] = phi * drift_e[i-1] + noise_e[i]
        drift_n[i] = phi * drift_n[i-1] + noise_n[i]
        
    # 2. Dynamic lag error proportional to turning acceleration
    turn_accel = np.gradient(heading_curve, dt_seconds) * speed_profile
    lag_e = 0.4 * turn_accel * np.cos(heading_curve)
    lag_n = -0.4 * turn_accel * np.sin(heading_curve)
    
    # 3. Urban canyon multipath bursts
    canyon_error_e = np.zeros(num_points)
    canyon_error_n = np.zeros(num_points)
    if urban_canyon:
        # Create 2 urban canyon sectors where satellite line-of-sight is blocked
        canyon_mask = ((t > 100) & (t < 200)) | ((t > 320) & (t < 420))
        canyon_error_e[canyon_mask] = np.random.normal(2.5, 1.2, np.sum(canyon_mask))
        canyon_error_n[canyon_mask] = np.random.normal(-3.0, 1.4, np.sum(canyon_mask))
        
    total_error_e = drift_e + lag_e + canyon_error_e
    total_error_n = drift_n + lag_n + canyon_error_n
    
    # Observed GPS positions = Ground Truth - Error (so GT = Obs + Error)
    x_obs = x_gt - total_error_e
    y_obs = y_gt - total_error_n
    
    lat_obs = base_lat + (y_obs / METERS_PER_DEG_LAT)
    lon_obs = base_lon + (x_obs / meters_per_deg_lon)
    
    df = pd.DataFrame({
        "tripId": "[SIMULATED] Synthetic_Urban_Route_1",
        "UnixTimeMillis": millis.astype(np.int64),
        "LatitudeDegrees_obs": lat_obs,
        "LongitudeDegrees_obs": lon_obs,
        "LatitudeDegrees_gt": lat_gt,
        "LongitudeDegrees_gt": lon_gt,
        "is_simulated": True,
    })
    return df
