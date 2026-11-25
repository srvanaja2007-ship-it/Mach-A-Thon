"""
Configuration and constants for AI-Based GPS/GNSS Position Error Correction System.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Raw files
GT_RAW_FILE = RAW_DATA_DIR / "ground_truths_train.csv"
OBS_RAW_FILE = RAW_DATA_DIR / "baseline_locations_train.csv"

# Processed files
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "gsdc_train.parquet"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "gsdc_test.parquet"
TRAIN_DATA_CSV = PROCESSED_DATA_DIR / "gsdc_train.csv"
TEST_DATA_CSV = PROCESSED_DATA_DIR / "gsdc_test.csv"

# WGS84 Geodetic Constants
EARTH_RADIUS_METERS = 6371000.0  # Mean Earth radius in meters
WGS84_A = 6378137.0             # Semi-major axis (meters)
WGS84_B = 6356752.314245        # Semi-minor axis (meters)
METERS_PER_DEG_LAT = 111139.0   # Approximate meters per degree latitude

# Model Feature Configuration
FEATURE_COLUMNS = [
    "speed",
    "accel",
    "heading",
    "turn_rate",
    "dx",
    "dy",
    "dx_lag1",
    "dy_lag1",
    "dx_lag2",
    "dy_lag2",
    "res_x",
    "res_y",
    "speed_roll_mean_3",
    "speed_roll_std_3",
    "turn_roll_std_3",
]

TARGET_COLUMNS = ["target_err_N", "target_err_E"]

# Kinematic Physical Bounds
# Maximum realistic correction offset in meters per epoch (guards against model extrapolation on multipath bursts)
MAX_PHYSICAL_CORRECTION_METERS = 5.0

# Model Hyperparameters
XGB_PARAMS = {
    "n_estimators": 120,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": -1,
}

RF_PARAMS = {
    "n_estimators": 60,
    "max_depth": 12,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1,
}
