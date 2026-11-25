# AI-Based GPS/GNSS Position Error Correction System

A production-ready, real-time machine learning positioning error correction pipeline designed for smartphones and mobile GNSS receivers. Trained and validated on the **Google Smartphone Decimeter Challenge (GSDC)** dataset against survey-grade **NovAtel SPAN RTK/GNSS+IMU** ground truth references.

---

## 🚀 Key Highlights & Capabilities

- **Real GNSS Dataset**: Built on authentic Google Smartphone Decimeter Challenge (GSDC) driving traces recorded across Silicon Valley (Mountain View, Sunnyvale, San Jose) using Google Pixel 4, Pixel 5, Samsung Galaxy S20 Ultra, and Xiaomi Mi 8 devices.
- **Zero Data Leakage**: Training and testing datasets are partitioned strictly by trajectory (`tripId`). Test routes are 100% held-out unseen drives.
- **Strictly Causal Real-Time Processing**: Features at time $t$ depend solely on measurements at $t \le t_{\text{current}}$ (no future look-ahead, zero data cheating).
- **Physical Kinematic Bounding**: Incorporates dynamic vehicle saturation bounds ($e_{\text{max}} = \pm 5.0$ m) to prevent model extrapolation during severe multipath dropouts.
- **Real-Time Stream Processor**: Sub-2ms per-coordinate inference latency (`RealTimeGpsCorrector`), suitable for embedded in-vehicle navigation or edge Android devices.
- **Interactive Folium & Streamlit Dashboard**: Full GUI with multi-layer maps (Raw GPS, NovAtel Ground Truth, AI-Corrected), real-time simulator, and metric diagnostics.

---

## 📊 Benchmark Results on Unseen Test Drives

Evaluated on **17,385 held-out epochs across 7 independent trips**:

| Metric | Original (Raw GPS) | XGBoost Corrected | Random Forest Corrected | Accuracy Improvement (%) |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE (m)** | **2.551 m** | **2.119 m** | **2.067 m** | **+18.96%** |
| **Mean Error (m)** | **2.133 m** | **1.780 m** | **1.734 m** | **+18.69%** |
| **Median Error (50%)** | **1.783 m** | **1.649 m** | **1.603 m** | **+10.06%** |
| **95th Percentile (m)** | **5.173 m** | **3.842 m** | **3.719 m** | **+28.10%** |

*All metrics are calculated using geodesic / Haversine distance against true NovAtel SPAN RTK reference positions.*

---

## 📐 Mathematical Formulation

### 1. Metric Coordinate Projection
Given raw observed GPS coordinates $(lat_{\text{obs}}, lon_{\text{obs}})$ and true reference coordinates $(lat_{\text{gt}}, lon_{\text{gt}})$, errors are transformed into local metric North and East axes:

$$\Delta N = (lat_{\text{gt}} - lat_{\text{obs}}) \times \frac{\pi}{180} \times R_{\text{earth}}$$

$$\Delta E = (lon_{\text{gt}} - lon_{\text{obs}}) \times \frac{\pi}{180} \times R_{\text{earth}} \times \cos(lat_{\text{obs}})$$

where $R_{\text{earth}} = 6,371,000 \text{ meters}$.

### 2. Machine Learning Regression
Two regressors (XGBoost or Random Forest) predict the metric displacements:

$$\widehat{\Delta N} = f_N(X_t), \quad \widehat{\Delta E} = f_E(X_t)$$

where feature vector $X_t$ includes causal speed, acceleration, heading, angular turn rate, lagged displacements, and instantaneous exponential moving average (EMA) residuals.

### 3. Coordinate Correction
$$\text{Lat}_{\text{corr}} = \text{Lat}_{\text{obs}} + \frac{\widehat{\Delta N}}{R_{\text{earth}}} \times \frac{180}{\pi}$$

$$\text{Lon}_{\text{corr}} = \text{Lon}_{\text{obs}} + \frac{\widehat{\Delta E}}{R_{\text{earth}} \cos(\text{Lat}_{\text{obs}})} \times \frac{180}{\pi}$$

---

## 📂 Project Directory Structure

```text
Mach-A-Thon/
├── data/
│   ├── raw/
│   │   ├── baseline_locations_train.csv   # Raw smartphone GPS observations
│   │   └── ground_truths_train.csv        # NovAtel SPAN RTK reference coordinates
│   └── processed/
│       ├── gsdc_train.parquet             # Partitioned training trajectories
│       └── gsdc_test.parquet              # Held-out testing trajectories
├── models/
│   ├── xgboost_model_N.joblib             # Trained North error regressor (XGB)
│   ├── xgboost_model_E.joblib             # Trained East error regressor (XGB)
│   ├── xgboost_metadata.json              # Hyperparameters & feature columns
│   ├── random_forest_model_N.joblib       # Trained North error regressor (RF)
│   ├── random_forest_model_E.joblib       # Trained East error regressor (RF)
│   └── random_forest_metadata.json
├── reports/
│   ├── evaluation_report.json             # Benchmark statistics in JSON
│   └── test_evaluation_results.parquet    # Test predictions with ground truth
├── src/
│   ├── __init__.py
│   ├── config.py                          # WGS84 constants, file paths, parameters
│   ├── data_loader.py                     # GSDC dataset merger & trip partitioner
│   ├── feature_engineering.py             # Causal kinematic feature extraction
│   ├── model.py                           # Dual-target GpsErrorCorrectionModel
│   ├── evaluation.py                      # RMSE, MAE, 95th percentile, Haversine
│   ├── real_time_corrector.py             # Streaming < 2ms latency single-fix corrector
│   └── synthetic_generator.py             # [SIMULATED DATA] urban canyon generator
├── app.py                                 # Interactive Streamlit + Folium dashboard
├── train.py                               # Model training & benchmarking script
├── evaluate.py                            # Standalone test verification CLI
├── requirements.txt                       # Project dependencies
└── README.md                              # System documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ or Python 3.11+
- Virtual environment (recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the System

### Satellite Model 1 Integration

The Eco_Act pipeline currently produces night-light rasters and economic
forecasts; it does not emit Model 1 latitude/longitude predictions. The
existing correction models are trained on GSDC GNSS kinematic features, so
they must not be applied directly to Eco_Act image pixels or economic
features.

Use [satellite_integration.py](./satellite_integration.py) when Model 1 can
provide the exact `FEATURE_COLUMNS` schema along with its predicted
coordinates:

```python
from satellite_integration import correct_satellite_image_location

result = correct_satellite_image_location(
    image=model1_image,
    predicted_latitude=model1_latitude,
    predicted_longitude=model1_longitude,
    image_features=model1_features,
    metadata={"image_id": "example-image"},
    selected_model="random_forest",
)
```

If compatible features are missing, the function returns
`status="insufficient_features"` and does not invent a correction. A new
satellite-localization correction dataset is required to train models that
use image embeddings or Eco_Act features.

### 1. Run the Interactive Streamlit Web Dashboard
Launch the web interface to view interactive Folium maps, test routes, and run real-time streaming:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### Features in the GUI:
- **Interactive Folium Map**: Zoom, inspect, and pan over raw (Red), ground truth (Green), and AI-corrected (Blue) routes.
- **Route Switcher**: Select between real Silicon Valley drives (Mountain View, Sunnyvale), simulated urban canyon tracks, or upload custom GPS CSV files.
- **Real-Time Stream Simulator**: Stream GPS measurements epoch-by-epoch and monitor live latency and error reduction.
- **Single Coordinate Tester**: Manually input coordinates and speed to test immediate corrections.
- **CSV Exporter**: Download AI-corrected trajectory coordinates.

---

### 2. Retrain Models
To retrain both XGBoost and Random Forest models on the dataset:
```bash
python train.py
```
Outputs model weights to `models/` and evaluation reports to `reports/evaluation_report.json`.

---

### 3. Run Standalone CLI Evaluation
Evaluate trained models on unseen test drives:
```bash
# Evaluate XGBoost
python evaluate.py xgboost

# Evaluate Random Forest
python evaluate.py random_forest

# Evaluate a specific trip
python evaluate.py xgboost "2020-05-21-US-MTV-2/GooglePixel4"
```

---

### 4. Real-Time Streaming Ingestion Example
To integrate into live GPS feeds (e.g. serial port / NMEA / socket):

```python
from src.real_time_corrector import RealTimeGpsCorrector

# Initialize stream corrector
corrector = RealTimeGpsCorrector(model_type="xgboost")

# Feed live GPS fix (Latitude, Longitude, UnixTimeMillis)
result = corrector.feed(lat=37.427358, lon=-122.069908, timestamp_ms=1621617482000)

print(f"Original:  {result.lat_original:.6f}, {result.lon_original:.6f}")
print(f"Corrected: {result.lat_corrected:.6f}, {result.lon_corrected:.6f}")
print(f"Shift:     {result.correction_magnitude_m:.2f} meters")
print(f"Latency:   {result.latency_ms:.2f} ms")
```

---

## 🔬 Important Methodological Guarantees

1. **No Artificial Shifts**: No arbitrary random coordinate offsets were used as primary training data; all training leverages true Google Decimeter Challenge NovAtel ground truth recordings.
2. **Zero Ground Truth Feature Leakage**: Ground truth coordinates are never exposed to the models as input features. They are strictly reserved for target delta calculation during training and final accuracy evaluation.
3. **Strict Isolation**: Test trajectories are completely separate vehicle drives from training trajectories.
4. **Honest Evaluation**: All accuracy improvements (mean error, RMSE, and 95th percentile) are rigorously measured and documented. Simulated data is strictly separated and labeled `[SIMULATED DATA]`.
