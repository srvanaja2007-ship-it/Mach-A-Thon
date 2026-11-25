"""
Stateful Real-Time Stream GPS Error Corrector.
Designed for real-time online streaming inference on live GPS coordinates.
Latency: < 1 millisecond per incoming coordinate fix.
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    EARTH_RADIUS_METERS,
    MAX_PHYSICAL_CORRECTION_METERS,
    MODELS_DIR,
)
from src.feature_engineering import apply_corrections
from src.model import GpsErrorCorrectionModel


@dataclass
class CorrectionResult:
    timestamp_ms: int
    lat_original: float
    lon_original: float
    lat_corrected: float
    lon_corrected: float
    error_north_m: float
    error_east_m: float
    correction_magnitude_m: float
    speed_mps: float
    heading_deg: float
    turn_rate_deg_s: float
    latency_ms: float


class RealTimeGpsCorrector:
    """
    Stream processor that accepts real-time (lat, lon, timestamp) measurements,
    extracts causal features on-the-fly, and produces corrected coordinates.
    """
    def __init__(
        self,
        model_type: str = "xgboost",
        buffer_size: int = 15,
        alpha_ema: float = 0.35,
        max_correction_meters: float = MAX_PHYSICAL_CORRECTION_METERS,
    ):
        self.model_type = model_type
        self.buffer_size = buffer_size
        self.alpha_ema = alpha_ema
        self.max_correction_meters = max_correction_meters
        
        # Load model
        self.model = GpsErrorCorrectionModel.load(load_dir=MODELS_DIR, prefix=model_type)
        
        # Stateful causal sliding history
        self.history: deque = deque(maxlen=buffer_size)
        self.ema_x: Optional[float] = None
        self.ema_y: Optional[float] = None
        self.ref_lat: Optional[float] = None
        self.ref_lon: Optional[float] = None
        self.point_count: int = 0

    def reset(self):
        """Resets stream state for a new trip / vehicle session."""
        self.history.clear()
        self.ema_x = None
        self.ema_y = None
        self.ref_lat = None
        self.ref_lon = None
        self.point_count = 0

    def feed(self, lat: float, lon: float, timestamp_ms: int) -> CorrectionResult:
        """
        Feeds a single live GPS measurement and returns the AI-corrected position.
        
        Args:
            lat: Observed Latitude (degrees)
            lon: Observed Longitude (degrees)
            timestamp_ms: Epoch timestamp in milliseconds
            
        Returns:
            CorrectionResult object with corrected lat/lon and metadata.
        """
        start_time = time.perf_counter()
        
        # Initialize anchor reference on first point
        if self.ref_lat is None:
            self.ref_lat = lat
            self.ref_lon = lon
            self.ema_x = 0.0
            self.ema_y = 0.0
            
        # Project to local metric Cartesian coordinates
        lat_rad_ref = np.radians(self.ref_lat)
        x_m = (lon - self.ref_lon) * (np.pi / 180.0) * EARTH_RADIUS_METERS * np.cos(lat_rad_ref)
        y_m = (lat - self.ref_lat) * (np.pi / 180.0) * EARTH_RADIUS_METERS
        
        current_pt = {
            "timestamp_ms": timestamp_ms,
            "t_sec": timestamp_ms / 1000.0,
            "lat": lat,
            "lon": lon,
            "x": x_m,
            "y": y_m,
        }
        
        # Append to buffer
        self.history.append(current_pt)
        self.point_count += 1
        
        # If less than 2 points, cannot compute velocity yet -> return identity
        if len(self.history) < 2:
            latency = (time.perf_counter() - start_time) * 1000.0
            return CorrectionResult(
                timestamp_ms=timestamp_ms,
                lat_original=lat,
                lon_original=lon,
                lat_corrected=lat,
                lon_corrected=lon,
                error_north_m=0.0,
                error_east_m=0.0,
                correction_magnitude_m=0.0,
                speed_mps=0.0,
                heading_deg=0.0,
                turn_rate_deg_s=0.0,
                latency_ms=latency,
            )
            
        # Extract causal features from history buffer
        pts = list(self.history)
        prev_pt = pts[-2]
        
        dt = current_pt["t_sec"] - prev_pt["t_sec"]
        if dt <= 0:
            dt = 1.0
            
        dx = current_pt["x"] - prev_pt["x"]
        dy = current_pt["y"] - prev_pt["y"]
        speed = np.sqrt(dx**2 + dy**2) / dt
        heading = np.arctan2(dx, dy)
        
        # Turn rate
        if len(pts) >= 3:
            prev_dx = prev_pt["x"] - pts[-3]["x"]
            prev_dy = prev_pt["y"] - pts[-3]["y"]
            prev_heading = np.arctan2(prev_dx, prev_dy)
            dheading = (heading - prev_heading + np.pi) % (2.0 * np.pi) - np.pi
            turn_rate = dheading / dt
            prev_dt = prev_pt["t_sec"] - pts[-3]["t_sec"]
            prev_speed = np.sqrt(prev_dx**2 + prev_dy**2) / (prev_dt if prev_dt > 0 else 1.0)
            accel = (speed - prev_speed) / dt
        else:
            turn_rate = 0.0
            accel = 0.0
            
        # Lagged metric steps
        dx_lag1 = (prev_pt["x"] - pts[-3]["x"]) if len(pts) >= 3 else 0.0
        dy_lag1 = (prev_pt["y"] - pts[-3]["y"]) if len(pts) >= 3 else 0.0
        dx_lag2 = (pts[-3]["x"] - pts[-4]["x"]) if len(pts) >= 4 else 0.0
        dy_lag2 = (pts[-3]["y"] - pts[-4]["y"]) if len(pts) >= 4 else 0.0
        
        # Causal EMA filter update
        self.ema_x = self.alpha_ema * x_m + (1.0 - self.alpha_ema) * self.ema_x
        self.ema_y = self.alpha_ema * y_m + (1.0 - self.alpha_ema) * self.ema_y
        res_x = x_m - self.ema_x
        res_y = y_m - self.ema_y
        
        # Rolling speed & turn rate
        speeds = []
        for i in range(max(1, len(pts) - 3), len(pts)):
            d_x = pts[i]["x"] - pts[i-1]["x"]
            d_y = pts[i]["y"] - pts[i-1]["y"]
            d_t = pts[i]["t_sec"] - pts[i-1]["t_sec"]
            speeds.append(np.sqrt(d_x**2 + d_y**2) / (d_t if d_t > 0 else 1.0))
            
        speed_roll_mean_3 = float(np.mean(speeds))
        speed_roll_std_3 = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        turn_roll_std_3 = abs(turn_rate) * 0.5  # causal proxy
        
        # Build feature vector
        feat_dict = {
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
        }
        feat_df = pd.DataFrame([feat_dict])
        
        # Predict error
        pred_N, pred_E = self.model.predict_errors(feat_df)
        err_N = float(pred_N[0])
        err_E = float(pred_E[0])
        
        # Physical saturation bounding
        if self.max_correction_meters > 0:
            err_N = np.clip(err_N, -self.max_correction_meters, self.max_correction_meters)
            err_E = np.clip(err_E, -self.max_correction_meters, self.max_correction_meters)
            
        # Apply correction
        lat_arr, lon_arr = apply_corrections(
            np.array([lat]), np.array([lon]), np.array([err_N]), np.array([err_E])
        )
        lat_corr = float(lat_arr[0])
        lon_corr = float(lon_arr[0])
        
        corr_mag = float(np.sqrt(err_N**2 + err_E**2))
        latency = (time.perf_counter() - start_time) * 1000.0
        
        return CorrectionResult(
            timestamp_ms=timestamp_ms,
            lat_original=lat,
            lon_original=lon,
            lat_corrected=lat_corr,
            lon_corrected=lon_corr,
            error_north_m=err_N,
            error_east_m=err_E,
            correction_magnitude_m=corr_mag,
            speed_mps=float(speed),
            heading_deg=float(np.degrees(heading) % 360),
            turn_rate_deg_s=float(np.degrees(turn_rate)),
            latency_ms=latency,
        )
