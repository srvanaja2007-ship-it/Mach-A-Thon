"""
Causal Feature Engineering for GNSS Positioning Error Correction.
Strictly causal: features at time t depend ONLY on measurements up to time t (no look-ahead).
Reference ground-truth coordinates are NEVER used as input features.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from src.config import EARTH_RADIUS_METERS


def compute_metric_displacements(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects geodetic latitude/longitude to local metric coordinates (East, North) in meters.
    """
    lat_rad = np.radians(lat_deg)
    ref_lat_rad = np.radians(ref_lat)
    
    # North coordinate (y) and East coordinate (x)
    y_m = (lat_deg - ref_lat) * (np.pi / 180.0) * EARTH_RADIUS_METERS
    x_m = (lon_deg - ref_lon) * (np.pi / 180.0) * EARTH_RADIUS_METERS * np.cos(ref_lat_rad)
    return x_m, y_m


def extract_features_single_trip(
    trip_df: pd.DataFrame,
    include_target: bool = True,
    alpha_ema: float = 0.35,
) -> pd.DataFrame:
    """
    Extracts causal kinematic and residual features for a single continuous trip.
    
    Features engineered (strictly causal):
    - speed (m/s)
    - accel (m/s^2)
    - heading (rad)
    - turn_rate (rad/s)
    - dx, dy (instantaneous metric delta)
    - dx_lag1, dy_lag1 (lag 1 step)
    - dx_lag2, dy_lag2 (lag 2 steps)
    - res_x, res_y (instantaneous deviation from causal exponential moving average)
    - speed_roll_mean_3, speed_roll_std_3 (rolling window over past 3 points)
    - turn_roll_std_3 (rolling turn rate variability over past 3 points)
    """
    df = trip_df.copy().sort_values("UnixTimeMillis").reset_index(drop=True)
    n = len(df)
    
    lat_obs = df["LatitudeDegrees_obs"].values
    lon_obs = df["LongitudeDegrees_obs"].values
    time_sec = df["UnixTimeMillis"].values / 1000.0
    
    # Local coordinate projection anchored at the first point of the trajectory
    x_m, y_m = compute_metric_displacements(lat_obs, lon_obs, lat_obs[0], lon_obs[0])
    
    # Time delta
    dt = np.diff(time_sec, prepend=time_sec[0] + 1.0)
    dt[dt <= 0] = 1.0  # guard against zero/negative dt
    
    # Metric velocity vectors
    dx = np.diff(x_m, prepend=x_m[0])
    dy = np.diff(y_m, prepend=y_m[0])
    
    speed = np.sqrt(dx**2 + dy**2) / dt
    
    # Heading angle from North: atan2(dx, dy)
    heading = np.arctan2(dx, dy)
    
    # Heading turn rate with angle wrapping [-pi, pi]
    dheading = np.diff(heading, prepend=heading[0])
    dheading = (dheading + np.pi) % (2.0 * np.pi) - np.pi
    turn_rate = dheading / dt
    
    # Acceleration
    accel = np.diff(speed, prepend=speed[0]) / dt
    
    # Lagged features (strictly causal)
    dx_lag1 = np.roll(dx, 1); dx_lag1[0] = 0.0
    dy_lag1 = np.roll(dy, 1); dy_lag1[0] = 0.0
    
    dx_lag2 = np.roll(dx, 2); dx_lag2[:2] = 0.0
    dy_lag2 = np.roll(dy, 2); dy_lag2[:2] = 0.0
    
    # Causal Exponential Moving Average (EMA) filter on trajectory
    # Filters out high-frequency multipath noise causally:
    # y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    ema_x = np.zeros(n)
    ema_y = np.zeros(n)
    ema_x[0] = x_m[0]
    ema_y[0] = y_m[0]
    for i in range(1, n):
        ema_x[i] = alpha_ema * x_m[i] + (1.0 - alpha_ema) * ema_x[i-1]
        ema_y[i] = alpha_ema * y_m[i] + (1.0 - alpha_ema) * ema_y[i-1]
        
    res_x = x_m - ema_x
    res_y = y_m - ema_y
    
    # Rolling causal statistics (window=3)
    s_series = pd.Series(speed)
    speed_roll_mean_3 = s_series.rolling(window=3, min_periods=1).mean().values
    speed_roll_std_3 = s_series.rolling(window=3, min_periods=1).std().fillna(0.0).values
    
    tr_series = pd.Series(turn_rate)
    turn_roll_std_3 = tr_series.rolling(window=3, min_periods=1).std().fillna(0.0).values
    
    features_df = pd.DataFrame({
        "speed": speed,
        "accel": accel,
        "heading": heading,
        "turn_rate": turn_rate,
        "dx": dx,
        "dy": dy,
        "dx_lag1": dx_lag1,
        "dy_lag1": dy_lag1,
        "dx_lag2": dx_lag2,
        "dy_lag2": dy_lag2,
        "res_x": res_x,
        "res_y": res_y,
        "speed_roll_mean_3": speed_roll_mean_3,
        "speed_roll_std_3": speed_roll_std_3,
        "turn_roll_std_3": turn_roll_std_3,
    })
    
    # Merge metadata
    result_df = pd.concat([
        df[["tripId", "UnixTimeMillis", "LatitudeDegrees_obs", "LongitudeDegrees_obs"]],
        features_df
    ], axis=1)
    
    if "is_simulated" in df.columns:
        result_df["is_simulated"] = df["is_simulated"]
        
    # Calculate ground truth target errors if available
    if include_target and "LatitudeDegrees_gt" in df.columns and "LongitudeDegrees_gt" in df.columns:
        lat_gt = df["LatitudeDegrees_gt"].values
        lon_gt = df["LongitudeDegrees_gt"].values
        
        # Target error in meters (North and East)
        # target_err_N = (lat_gt - lat_obs) in meters
        # target_err_E = (lon_gt - lon_obs) in meters
        lat_rad = np.radians(lat_obs)
        target_err_N = (lat_gt - lat_obs) * (np.pi / 180.0) * EARTH_RADIUS_METERS
        target_err_E = (lon_gt - lon_obs) * (np.pi / 180.0) * EARTH_RADIUS_METERS * np.cos(lat_rad)
        
        result_df["LatitudeDegrees_gt"] = lat_gt
        result_df["LongitudeDegrees_gt"] = lon_gt
        result_df["target_err_N"] = target_err_N
        result_df["target_err_E"] = target_err_E
        
    return result_df


def extract_features_dataset(
    df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Extracts causal features trip by trip across the full dataset.
    """
    processed_trips = []
    for trip_id, trip_group in df.groupby("tripId", sort=False):
        trip_feat = extract_features_single_trip(trip_group, include_target=include_target)
        processed_trips.append(trip_feat)
    return pd.concat(processed_trips, ignore_index=True)


def apply_corrections(
    lat_obs: np.ndarray,
    lon_obs: np.ndarray,
    pred_err_N: np.ndarray,
    pred_err_E: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies predicted metric North and East positioning errors to observed GPS coordinates.
    
    Since target_err_N = (lat_gt - lat_obs) in meters:
    lat_corrected = lat_obs + delta_lat_deg
    lon_corrected = lon_obs + delta_lon_deg
    """
    lat_rad = np.radians(lat_obs)
    dlat_deg = (pred_err_N / EARTH_RADIUS_METERS) * (180.0 / np.pi)
    dlon_deg = (pred_err_E / (EARTH_RADIUS_METERS * np.cos(lat_rad))) * (180.0 / np.pi)
    
    lat_corr = lat_obs + dlat_deg
    lon_corr = lon_obs + dlon_deg
    return lat_corr, lon_corr
