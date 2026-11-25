"""
AI-Based GPS/GNSS Position Error Correction System - Interactive Dashboard.
Features:
- Folium Interactive Map showing Raw GPS, Ground Truth, and AI-Corrected Positions
- Real-Time Streaming Simulator (< 2ms per fix)
- Comprehensive Metric Comparisons (RMSE, Mean, Median, 95th percentile, % improvement)
- Support for Real GSDC Trajectories, Simulated Urban Canyon Routes, and Custom CSV Uploads
"""
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px

# Streamlit Folium bridge
try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    HAS_STREAMLIT_FOLIUM = False

from src.config import (
    FEATURE_COLUMNS,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    MODELS_DIR,
    MAX_PHYSICAL_CORRECTION_METERS,
)
from src.model import GpsErrorCorrectionModel
from src.feature_engineering import extract_features_single_trip, extract_features_dataset
from src.evaluation import (
    evaluate_positioning_accuracy,
    compare_models_on_test_data,
    geodesic_distance_wgs84_vectorized,
)
from src.synthetic_generator import generate_synthetic_trajectory
from src.real_time_corrector import RealTimeGpsCorrector

# Page configuration
st.set_page_config(
    page_title="AI GPS Error Correction System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .metric-badge-good {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .metric-badge-neutral {
        display: inline-block;
        background: #F1F5F9;
        color: #475569;
        font-size: 0.82rem;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str) -> GpsErrorCorrectionModel:
    """Loads and caches the specified ML model."""
    return GpsErrorCorrectionModel.load(load_dir=MODELS_DIR, prefix=model_name)


@st.cache_data
def load_test_dataset() -> pd.DataFrame:
    """Loads pre-cached test trajectories."""
    if TEST_DATA_FILE.exists():
        return pd.read_parquet(TEST_DATA_FILE)
    return pd.DataFrame()


@st.cache_data
def load_train_dataset() -> pd.DataFrame:
    """Loads pre-cached train trajectories for leakage auditing."""
    if TRAIN_DATA_FILE.exists():
        return pd.read_parquet(TRAIN_DATA_FILE)
    return pd.DataFrame()


@st.cache_data
def get_heldout_benchmark(_rf_model: GpsErrorCorrectionModel, _xgb_model: GpsErrorCorrectionModel, max_bound: float):
    """
    Evaluates both trained models on the held-out unseen GSDC test dataset
    to automatically identify the better-performing model.
    """
    test_all = load_test_dataset()
    train_all = load_train_dataset()
    if test_all.empty:
        return None
    eval_feat = extract_features_dataset(test_all, include_target=False)
    eval_feat["LatitudeDegrees_gt"] = test_all["LatitudeDegrees_gt"].to_numpy(dtype=float)
    eval_feat["LongitudeDegrees_gt"] = test_all["LongitudeDegrees_gt"].to_numpy(dtype=float)
    return compare_models_on_test_data(
        test_feat_df=eval_feat,
        rf_model=_rf_model,
        xgb_model=_xgb_model,
        feature_columns=FEATURE_COLUMNS,
        max_correction_meters=max_bound,
        train_df=train_all if not train_all.empty else None,
    )


def render_folium_map(
    df: pd.DataFrame,
    show_raw: bool = True,
    show_gt: bool = True,
    show_rf: bool = True,
    show_xgb: bool = True,
    show_vectors: bool = False,
    active_model_type: str = "random_forest",
    sample_step: int = 1,
):
    """Generates an interactive Folium map with Raw, Ground Truth, Random Forest, and XGBoost layers."""
    plot_df = df.iloc[::sample_step].reset_index(drop=True)
    
    # Calculate map center
    center_lat = float(plot_df["LatitudeDegrees_obs"].mean())
    center_lon = float(plot_df["LongitudeDegrees_obs"].mean())
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    
    # Coordinates (drop any NaNs to keep Folium PolyLine safe)
    plot_df = plot_df.dropna(subset=["LatitudeDegrees_obs", "LongitudeDegrees_obs"]).reset_index(drop=True)
    if plot_df.empty:
        return folium.Map(location=[37.4221, -122.0841], zoom_start=15, tiles="OpenStreetMap")

    pts_obs = plot_df[["LatitudeDegrees_obs", "LongitudeDegrees_obs"]].values.tolist()
    has_gt = "LatitudeDegrees_gt" in plot_df.columns and plot_df["LatitudeDegrees_gt"].notna().all()
    has_rf = "lat_corr_rf" in plot_df.columns and plot_df["lat_corr_rf"].notna().all()
    has_xgb = "lat_corr_xgb" in plot_df.columns and plot_df["lat_corr_xgb"].notna().all()
    
    if has_gt:
        pts_gt = plot_df[["LatitudeDegrees_gt", "LongitudeDegrees_gt"]].values.tolist()
    if has_rf:
        pts_rf = plot_df[["lat_corr_rf", "lon_corr_rf"]].values.tolist()
    if has_xgb:
        pts_xgb = plot_df[["lat_corr_xgb", "lon_corr_xgb"]].values.tolist()
        
    # 1. Reference Ground Truth Layer (Green)
    if show_gt and has_gt:
        folium.PolyLine(
            pts_gt,
            color="#10B981",
            weight=4,
            opacity=0.9,
            tooltip="Reference Ground Truth (NovAtel SPAN RTK)",
        ).add_to(m)
        
    # 2. Raw Observed GPS Layer (Red)
    if show_raw:
        folium.PolyLine(
            pts_obs,
            color="#EF4444",
            weight=3,
            opacity=0.75,
            dash_array="5, 5",
            tooltip="Raw Observed GPS Track",
        ).add_to(m)
        
    # 3. Random Forest Corrected Layer (Purple)
    if show_rf and has_rf:
        folium.PolyLine(
            pts_rf,
            color="#8B5CF6",
            weight=3.5,
            opacity=0.88,
            tooltip="Random Forest Corrected Track",
        ).add_to(m)

    # 4. XGBoost Corrected Layer (Blue)
    if show_xgb and has_xgb:
        folium.PolyLine(
            pts_xgb,
            color="#2563EB",
            weight=3.5,
            opacity=0.88,
            tooltip="XGBoost Corrected Track",
        ).add_to(m)
        
    # Start and End Markers
    folium.Marker(
        pts_obs[0],
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        popup="Trip Origin",
    ).add_to(m)
    
    folium.Marker(
        pts_obs[-1],
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        popup="Trip Destination",
    ).add_to(m)
    
    # Sample markers with interactive tooltips showing both models' predictions
    marker_step = max(1, len(plot_df) // 25)
    for i in range(0, len(plot_df), marker_step):
        row = plot_df.iloc[i]
        raw_pt = [row["LatitudeDegrees_obs"], row["LongitudeDegrees_obs"]]
        
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 250px;">
            <b>Epoch #{i * sample_step}</b><br>
            <span style="color:#EF4444;">● Raw:</span> {row['LatitudeDegrees_obs']:.6f}, {row['LongitudeDegrees_obs']:.6f}<br>
        """
        if has_rf:
            popup_html += f"""<span style="color:#8B5CF6;">● RF:</span> {row['lat_corr_rf']:.6f}, {row['lon_corr_rf']:.6f}<br>"""
        if has_xgb:
            popup_html += f"""<span style="color:#2563EB;">● XGB:</span> {row['lat_corr_xgb']:.6f}, {row['lon_corr_xgb']:.6f}<br>"""
        if has_gt:
            popup_html += f"""<span style="color:#10B981;">● True:</span> {row['LatitudeDegrees_gt']:.6f}, {row['LongitudeDegrees_gt']:.6f}<br>"""
        if "error_raw_m" in row and "error_rf_m" in row and "error_xgb_m" in row:
            raw_e = row["error_raw_m"]
            rf_e = row["error_rf_m"]
            xgb_e = row["error_xgb_m"]
            popup_html += f"""
            <hr style="margin:4px 0;">
            <b>Raw Error:</b> {raw_e:.2f} m<br>
            <b>RF Error:</b> {rf_e:.2f} m ({raw_e - rf_e:+.2f} m)<br>
            <b>XGB Error:</b> {xgb_e:.2f} m ({raw_e - xgb_e:+.2f} m)
            """
        popup_html += "</div>"
        
        if show_raw:
            folium.CircleMarker(
                raw_pt, radius=4, color="#EF4444", fill=True, fill_color="#EF4444",
                fill_opacity=0.8, popup=folium.Popup(popup_html, max_width=270),
            ).add_to(m)
            
        if show_rf and has_rf:
            folium.CircleMarker(
                [row["lat_corr_rf"], row["lon_corr_rf"]],
                radius=4, color="#8B5CF6", fill=True, fill_color="#8B5CF6",
                fill_opacity=0.8, popup=folium.Popup(popup_html, max_width=270),
            ).add_to(m)

        if show_xgb and has_xgb:
            folium.CircleMarker(
                [row["lat_corr_xgb"], row["lon_corr_xgb"]],
                radius=4, color="#2563EB", fill=True, fill_color="#2563EB",
                fill_opacity=0.8, popup=folium.Popup(popup_html, max_width=270),
            ).add_to(m)
            
        if show_vectors:
            vec_lat = row["lat_corr_rf"] if active_model_type == "random_forest" and has_rf else row.get("lat_corr_xgb")
            vec_lon = row["lon_corr_rf"] if active_model_type == "random_forest" and has_rf else row.get("lon_corr_xgb")
            if vec_lat is not None and vec_lon is not None:
                folium.PolyLine(
                    [raw_pt, [vec_lat, vec_lon]],
                    color="#F59E0B", weight=2, opacity=0.75, dash_array="2, 4",
                ).add_to(m)
            
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position="topright", primary_length_unit="meters").add_to(m)
    
    all_lats = [p[0] for p in pts_obs]
    all_lons = [p[1] for p in pts_obs]
    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
    
    return m


def main():
    # Header
    st.markdown('<div class="main-title">🛰️ AI-Based GPS/GNSS Position Error Correction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Dual-Model Comparison (Random Forest vs XGBoost) — Real-time ML positioning error '
        'correction pipeline trained on <b>Google Smartphone Decimeter Challenge (GSDC)</b> ground truth benchmark.</div>',
        unsafe_allow_html=True,
    )
    
    # Load both trained models simultaneously (no retraining)
    try:
        rf_model = load_model("random_forest")
        xgb_model = load_model("xgboost")
    except Exception as e:
        st.error(f"Error loading trained models: {e}. Please ensure both Random Forest and XGBoost models exist in `{MODELS_DIR}`.")
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        max_bound = st.slider(
            "Max Physical Correction Bound (meters)",
            min_value=1.0,
            max_value=12.0,
            value=float(MAX_PHYSICAL_CORRECTION_METERS),
            step=0.5,
            help="Saturates single-epoch corrections to prevent divergence on severe multipath anomalies.",
        )

        # Automatically identify better-performing model on held-out test data
        heldout_bench = get_heldout_benchmark(rf_model, xgb_model, max_bound)
        if heldout_bench is not None:
            recommended_name = heldout_bench["better_model"]
            rf_label = "Random Forest Regressor (Recommended)" if recommended_name == "Random Forest" else "Random Forest Regressor"
            xgb_label = "XGBoost Regressor (Recommended)" if recommended_name == "XGBoost" else "XGBoost Regressor"
            default_idx = 0 if recommended_name == "Random Forest" else 1
        else:
            recommended_name = "Random Forest"
            rf_label = "Random Forest Regressor (Recommended)"
            xgb_label = "XGBoost Regressor"
            default_idx = 0

        model_choice = st.selectbox(
            "Primary / Active Model",
            options=[rf_label, xgb_label],
            index=default_idx,
            help="Auto-selected from held-out test RMSE. Both models still run simultaneously for comparison.",
        )
        model_type = "random_forest" if "Random Forest" in model_choice else "xgboost"
        model = rf_model if model_type == "random_forest" else xgb_model
        
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Select Trajectory Source",
            options=[
                "Real GSDC Test Trajectory",
                "Simulated Urban Canyon Route [SIMULATED]",
                "Upload Custom GPS CSV",
            ],
            index=0,
        )
        
        st.subheader("🗺️ Map Layer Visibility")
        show_raw = st.checkbox("Show Raw GPS (Red)", value=True)
        show_rf = st.checkbox("Show Random Forest (Purple)", value=True)
        show_xgb = st.checkbox("Show XGBoost (Blue)", value=True)
        show_gt = st.checkbox("Show Ground Truth (Green)", value=True)
        show_vectors = st.checkbox("Show Correction Vectors", value=False)
        sample_step = st.slider("Map Point Downsampling", min_value=1, max_value=5, value=1)
        
        st.markdown("---")
        st.markdown(
            f"**Recommended Model**: 🏆 `{recommended_name}`  \n"
            "**Inference Mode**: Simultaneous (RF + XGBoost)  \n"
            "**Input Features**: Causal Kinematics (No GT)"
        )

    # Load / Prepare Trajectory Data
    current_df = None
    trajectory_label = ""
    is_simulated = False
    
    if data_source == "Real GSDC Test Trajectory":
        test_raw = load_test_dataset()
        if test_raw.empty:
            st.warning("Processed test dataset not found. Generating default test trajectory...")
            current_df = generate_synthetic_trajectory(num_points=300)
            trajectory_label = "Synthetic Fallback Trajectory [SIMULATED]"
            is_simulated = True
        else:
            trips = test_raw["tripId"].unique().tolist()
            selected_trip = st.selectbox(
                "Select GSDC Test Drive Trajectory",
                options=trips,
                index=0,
                help="Actual smartphone vehicle trajectories recorded in Mountain View and Sunnyvale, CA.",
            )
            current_df = test_raw[test_raw["tripId"] == selected_trip].copy().reset_index(drop=True)
            trajectory_label = f"GSDC Real Drive: {selected_trip}"
            is_simulated = False

    elif data_source == "Simulated Urban Canyon Route [SIMULATED]":
        is_simulated = True
        n_pts = st.slider("Number of Trajectory Points", 100, 1000, 400, 50)
        canyon = st.checkbox("Inject Urban Canyon Multipath Bursts", value=True)
        current_df = generate_synthetic_trajectory(num_points=n_pts, urban_canyon=canyon)
        trajectory_label = "Synthetic Urban Canyon Trajectory [SIMULATED DATA]"
        st.info("⚠️ **Note**: This dataset is **SIMULATED** with synthetic multipath and Kalman lag to demonstrate urban canyon performance.")

    elif data_source == "Upload Custom GPS CSV":
        uploaded_file = st.file_uploader(
            "Upload GPS CSV File",
            type=["csv"],
            help="Must include 'Latitude' (or lat) and 'Longitude' (or lon) columns. Optional: 'UnixTimeMillis', 'Latitude_gt', 'Longitude_gt'.",
        )
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            # Normalize column names
            col_map = {}
            for c in user_df.columns:
                cl = c.lower().strip()
                is_gt_col = any(tok in cl for tok in ["gt", "true", "ref", "reference"])
                if "lat" in cl and not is_gt_col and "LatitudeDegrees_obs" not in col_map.values():
                    col_map[c] = "LatitudeDegrees_obs"
                elif "lon" in cl and not is_gt_col and "LongitudeDegrees_obs" not in col_map.values():
                    col_map[c] = "LongitudeDegrees_obs"
                elif "time" in cl and "UnixTimeMillis" not in col_map.values():
                    col_map[c] = "UnixTimeMillis"
                elif "lat" in cl and is_gt_col and "LatitudeDegrees_gt" not in col_map.values():
                    col_map[c] = "LatitudeDegrees_gt"
                elif "lon" in cl and is_gt_col and "LongitudeDegrees_gt" not in col_map.values():
                    col_map[c] = "LongitudeDegrees_gt"
            user_df = user_df.rename(columns=col_map)
            
            if "LatitudeDegrees_obs" in user_df.columns and "LongitudeDegrees_obs" in user_df.columns:
                user_df = user_df.dropna(subset=["LatitudeDegrees_obs", "LongitudeDegrees_obs"]).reset_index(drop=True)
                if "UnixTimeMillis" not in user_df.columns:
                    user_df["UnixTimeMillis"] = np.arange(len(user_df)) * 1000
                if "tripId" not in user_df.columns:
                    user_df["tripId"] = "Uploaded_Trajectory"
                current_df = user_df
                trajectory_label = f"Uploaded File: {uploaded_file.name}"
            else:
                st.error("Uploaded CSV must contain columns for Latitude and Longitude.")
                return
        else:
            st.info("Please upload a CSV file or select a Real GSDC Trajectory.")
            return

    if current_df is None or len(current_df) == 0:
        st.warning("No trajectory data available.")
        return

    # Extract causal inference features ONLY from observed GPS (no ground truth used during inference)
    with st.spinner("Extracting causal kinematic features and running Random Forest & XGBoost simultaneously..."):
        sorted_raw = current_df.sort_values("UnixTimeMillis").reset_index(drop=True)
        feat_df = extract_features_single_trip(sorted_raw, include_target=False)
        
        lat_obs = feat_df["LatitudeDegrees_obs"].to_numpy(dtype=float)
        lon_obs = feat_df["LongitudeDegrees_obs"].to_numpy(dtype=float)
        X_inference = feat_df.loc[:, FEATURE_COLUMNS]

        # Run BOTH models independently on identical GPS input features (never averaged)
        lat_rf, lon_rf, pred_N_rf, pred_E_rf = rf_model.correct_coordinates(
            lat_obs=lat_obs,
            lon_obs=lon_obs,
            X=X_inference,
            max_correction_meters=max_bound,
        )
        lat_xgb, lon_xgb, pred_N_xgb, pred_E_xgb = xgb_model.correct_coordinates(
            lat_obs=lat_obs,
            lon_obs=lon_obs,
            X=X_inference,
            max_correction_meters=max_bound,
        )

        feat_df["lat_corr_rf"] = lat_rf
        feat_df["lon_corr_rf"] = lon_rf
        feat_df["pred_err_N_rf"] = pred_N_rf
        feat_df["pred_err_E_rf"] = pred_E_rf

        feat_df["lat_corr_xgb"] = lat_xgb
        feat_df["lon_corr_xgb"] = lon_xgb
        feat_df["pred_err_N_xgb"] = pred_N_xgb
        feat_df["pred_err_E_xgb"] = pred_E_xgb

        # Active model alias (either RF or XGBoost, never combined)
        if model_type == "random_forest":
            lat_corr, lon_corr, pred_N, pred_E = lat_rf, lon_rf, pred_N_rf, pred_E_rf
        else:
            lat_corr, lon_corr, pred_N, pred_E = lat_xgb, lon_xgb, pred_N_xgb, pred_E_xgb

        feat_df["lat_corr"] = lat_corr
        feat_df["lon_corr"] = lon_corr
        feat_df["pred_err_N"] = pred_N
        feat_df["pred_err_E"] = pred_E
        
        has_ground_truth = "LatitudeDegrees_gt" in sorted_raw.columns and "LongitudeDegrees_gt" in sorted_raw.columns
        eval_metrics = None
        if has_ground_truth:
            lat_gt = sorted_raw["LatitudeDegrees_gt"].to_numpy(dtype=float)
            lon_gt = sorted_raw["LongitudeDegrees_gt"].to_numpy(dtype=float)
            feat_df["LatitudeDegrees_gt"] = lat_gt
            feat_df["LongitudeDegrees_gt"] = lon_gt

            raw_errs = geodesic_distance_wgs84_vectorized(lat_obs, lon_obs, lat_gt, lon_gt)
            rf_errs = geodesic_distance_wgs84_vectorized(lat_rf, lon_rf, lat_gt, lon_gt)
            xgb_errs = geodesic_distance_wgs84_vectorized(lat_xgb, lon_xgb, lat_gt, lon_gt)

            feat_df["error_raw_m"] = raw_errs
            feat_df["error_rf_m"] = rf_errs
            feat_df["error_xgb_m"] = xgb_errs

            eval_metrics = evaluate_positioning_accuracy(
                lat_obs, lon_obs, lat_gt, lon_gt, lat_corr, lon_corr
            )
            feat_df["error_corr_m"] = eval_metrics["corr_errors"]

    # Trajectory Info Banner
    badge_type = "🔴 [SIMULATED DATA]" if is_simulated else "🟢 [REAL GSDC DATA]"
    st.markdown(
        f"**Active Route:** `{trajectory_label}` | **Epochs:** `{len(feat_df):,}` points | **Type:** {badge_type}"
    )

    # KPI Metric Cards — Raw / Random Forest / XGBoost side-by-side (if ground truth available)
    if has_ground_truth and "error_raw_m" in feat_df.columns and "error_rf_m" in feat_df.columns and "error_xgb_m" in feat_df.columns:
        raw_arr = feat_df["error_raw_m"].to_numpy(dtype=float)
        rf_arr  = feat_df["error_rf_m"].to_numpy(dtype=float)
        xgb_arr = feat_df["error_xgb_m"].to_numpy(dtype=float)

        def _stats(arr):
            return {
                "rmse":   float(np.sqrt(np.mean(arr ** 2))),
                "mean":   float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p95":    float(np.percentile(arr, 95)),
            }

        raw_s = _stats(raw_arr)
        rf_s  = _stats(rf_arr)
        xgb_s = _stats(xgb_arr)

        def _improv(raw_val, model_val):
            if raw_val == 0:
                return 0.0
            return (raw_val - model_val) / raw_val * 100.0

        # -- Row 1 header --
        h0, h1, h2, h3, h4, h5 = st.columns([1.6, 1, 1, 1, 1, 1])
        h0.markdown("**Model**")
        h1.markdown("**RMSE (m)**")
        h2.markdown("**MAE (m)**")
        h3.markdown("**Median (m)**")
        h4.markdown("**95th Pct (m)**")
        h5.markdown("**RMSE Improv %**")

        # -- Raw GPS row --
        r0, r1, r2, r3, r4, r5 = st.columns([1.6, 1, 1, 1, 1, 1])
        r0.markdown("<span style='color:#EF4444;font-weight:700;'>● Raw GPS</span>", unsafe_allow_html=True)
        r1.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{raw_s['rmse']:.3f}</div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{raw_s['mean']:.3f}</div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{raw_s['median']:.3f}</div>", unsafe_allow_html=True)
        r4.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{raw_s['p95']:.3f}</div>", unsafe_allow_html=True)
        r5.markdown("<span class='metric-badge-neutral'>baseline</span>", unsafe_allow_html=True)

        # -- Random Forest row --
        rf_rmse_pct = _improv(raw_s["rmse"], rf_s["rmse"])
        b0, b1, b2, b3, b4, b5 = st.columns([1.6, 1, 1, 1, 1, 1])
        b0.markdown("<span style='color:#8B5CF6;font-weight:700;'>● Random Forest</span>", unsafe_allow_html=True)
        b1.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{rf_s['rmse']:.3f}</div>", unsafe_allow_html=True)
        b2.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{rf_s['mean']:.3f}</div>", unsafe_allow_html=True)
        b3.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{rf_s['median']:.3f}</div>", unsafe_allow_html=True)
        b4.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{rf_s['p95']:.3f}</div>", unsafe_allow_html=True)
        badge_rf = "metric-badge-good" if rf_rmse_pct > 0 else "metric-badge-neutral"
        b5.markdown(f"<span class='{badge_rf}'>{rf_rmse_pct:+.1f}%</span>", unsafe_allow_html=True)

        # -- XGBoost row --
        xgb_rmse_pct = _improv(raw_s["rmse"], xgb_s["rmse"])
        x0, x1, x2, x3, x4, x5 = st.columns([1.6, 1, 1, 1, 1, 1])
        x0.markdown("<span style='color:#2563EB;font-weight:700;'>● XGBoost</span>", unsafe_allow_html=True)
        x1.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{xgb_s['rmse']:.3f}</div>", unsafe_allow_html=True)
        x2.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{xgb_s['mean']:.3f}</div>", unsafe_allow_html=True)
        x3.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{xgb_s['median']:.3f}</div>", unsafe_allow_html=True)
        x4.markdown(f"<div class='metric-val' style='font-size:1.1rem;'>{xgb_s['p95']:.3f}</div>", unsafe_allow_html=True)
        badge_xgb = "metric-badge-good" if xgb_rmse_pct > 0 else "metric-badge-neutral"
        x5.markdown(f"<span class='{badge_xgb}'>{xgb_rmse_pct:+.1f}%</span>", unsafe_allow_html=True)

        st.write("")

    # Main Tabs
    tab_map, tab_compare, tab_analytics, tab_stream, tab_single, tab_data = st.tabs([
        "🗺️ Interactive Folium Map",
        "🏆 Model Comparison (RF vs XGBoost)",
        "📊 Accuracy & Diagnostics",
        "⚡ Real-Time Stream Simulator",
        "📍 Single Coordinate Correction",
        "📄 Data & Export",
    ])

    # Tab 1: Interactive Folium Map
    with tab_map:
        st.subheader("Interactive Trajectory Map (Raw vs Random Forest vs XGBoost vs Ground Truth)")
        map_obj = render_folium_map(
            feat_df,
            show_raw=show_raw,
            show_gt=show_gt,
            show_rf=show_rf,
            show_xgb=show_xgb,
            show_vectors=show_vectors,
            active_model_type=model_type,
            sample_step=sample_step,
        )
        
        # Legend guide
        st.markdown("""
        <div style="background:#F1F5F9; padding:8px 16px; border-radius:6px; font-size:13px; margin-bottom:12px;">
            <span style="color:#EF4444; font-weight:bold;">━━━ Raw GPS Track</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color:#8B5CF6; font-weight:bold;">━━━ Random Forest Corrected</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color:#2563EB; font-weight:bold;">━━━ XGBoost Corrected</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color:#10B981; font-weight:bold;">━━━ Ground Truth Reference</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color:#F59E0B; font-weight:bold;">╌╌╌ Correction Vectors</span>
        </div>
        """, unsafe_allow_html=True)
        
        if HAS_STREAMLIT_FOLIUM:
            st_folium(map_obj, use_container_width=True, height=560, returned_objects=[])
        else:
            map_html = map_obj._repr_html_()
            components.html(map_html, height=560)

    # Tab 2: Head-to-Head Model Comparison (RF vs XGBoost)
    with tab_compare:
        st.subheader("Real-World GPS Model Accuracy Evaluation (WGS84 Geodesic Distance)")
        eval_scope = st.radio(
            "Evaluation Scope",
            options=["Current Selected Trajectory", "All Held-Out GSDC Test Trajectories"],
            horizontal=True,
        )

        # Verify both trained models exist before running comparison
        missing_models = []
        for prefix in ("random_forest", "xgboost"):
            for suffix in ("metadata.json", "model_N.joblib", "model_E.joblib"):
                p = MODELS_DIR / f"{prefix}_{suffix}"
                if not p.exists():
                    missing_models.append(str(p))

        if missing_models:
            st.error(
                "Cannot run head-to-head comparison because trained model files are missing:\n"
                + "\n".join(f"- `{m}`" for m in missing_models)
            )
        else:
            rf_model_cmp = load_model("random_forest")
            xgb_model_cmp = load_model("xgboost")
            train_df_audit = load_train_dataset()

            if eval_scope == "All Held-Out GSDC Test Trajectories":
                test_all = load_test_dataset()
                if test_all.empty:
                    st.error(f"Missing GSDC test dataset at `{TEST_DATA_FILE}`.")
                    comp_res = None
                else:
                    eval_feat_df = extract_features_dataset(test_all, include_target=True)
                    try:
                        comp_res = compare_models_on_test_data(
                            test_feat_df=eval_feat_df,
                            rf_model=rf_model_cmp,
                            xgb_model=xgb_model_cmp,
                            feature_columns=FEATURE_COLUMNS,
                            max_correction_meters=max_bound,
                            train_df=train_df_audit if not train_df_audit.empty else None,
                        )
                    except ValueError as err:
                        st.error(f"Missing required ground-truth or coordinate data: {err}")
                        comp_res = None
            else:
                if not has_ground_truth:
                    st.warning(
                        "Ground-truth coordinates (`LatitudeDegrees_gt`, `LongitudeDegrees_gt`) are missing "
                        "for the current trajectory. Please select a GSDC test trajectory or upload a CSV with ground truth."
                    )
                    comp_res = None
                else:
                    try:
                        comp_res = compare_models_on_test_data(
                            test_feat_df=feat_df,
                            rf_model=rf_model_cmp,
                            xgb_model=xgb_model_cmp,
                            feature_columns=FEATURE_COLUMNS,
                            max_correction_meters=max_bound,
                            train_df=train_df_audit if not train_df_audit.empty else None,
                        )
                    except ValueError as err:
                        st.error(f"Missing required ground-truth or coordinate data: {err}")
                        comp_res = None

            if comp_res is not None:
                st.info(
                    "**Evaluation split:** Held-out GSDC test data only. "
                    "Both models received the same causal observed-GPS feature vector; "
                    "ground-truth coordinates were used only to score predictions."
                )
                # Top summary KPI row
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Test Observations", f"{comp_res['n_observations']:,}")
                k2.metric("Test Trajectories", f"{comp_res['n_trajectories']}")
                k3.metric("Raw GPS RMSE", f"{comp_res['raw_metrics']['rmse']:.4f} m")
                k4.metric(
                    "Random Forest RMSE",
                    f"{comp_res['rf_metrics']['rmse']:.4f} m",
                    f"{comp_res['rf_rmse_improvement_pct']:+.2f}%",
                )
                k5.metric(
                    "XGBoost RMSE",
                    f"{comp_res['xgb_metrics']['rmse']:.4f} m",
                    f"{comp_res['xgb_rmse_improvement_pct']:+.2f}%",
                )

                # Winner & Data Leakage Audit banner
                leak = comp_res["leakage_audit"]
                leak_status = "PASSED (0 overlapping trips, 0 ground-truth features)" if leak["passed"] else "FAILED"
                st.success(
                    f"**Better Performing Model:** {comp_res['better_model']} &nbsp;|&nbsp; "
                    f"**Data Leakage Audit:** {leak_status}"
                )

                # Comparison Table
                st.markdown("##### Side-by-Side Model Accuracy Comparison (WGS84 Geodesic Distance)")
                st.dataframe(comp_res["comparison_table"], width="stretch", hide_index=True)

                # Download comparison CSV
                cmp_csv = comp_res["comparison_table"].to_csv(index=False)
                st.download_button(
                    label="📥 Download Model Comparison CSV",
                    data=cmp_csv,
                    file_name="model_comparison_results.csv",
                    mime="text/csv",
                )

                # Charts: Grouped Bar Chart + Overlaid Error CDF
                ch1, ch2 = st.columns(2)
                with ch1:
                    comp_tbl = comp_res["comparison_table"]
                    metrics_to_plot = ["MAE (m)", "RMSE (m)", "Median Error (m)", "95th Percentile Error (m)"]
                    fig_bar = go.Figure()
                    colors = {"Raw GPS (Observed)": "#EF4444", "Random Forest": "#10B981", "XGBoost": "#2563EB"}
                    for _, r in comp_tbl.iterrows():
                        fig_bar.add_trace(go.Bar(
                            name=r["Model"],
                            x=metrics_to_plot,
                            y=[r[m] for m in metrics_to_plot],
                            marker_color=colors.get(r["Model"], "#64748B"),
                            text=[f"{r[m]:.2f}" for m in metrics_to_plot],
                            textposition="auto",
                        ))
                    fig_bar.update_layout(
                        title="Error Metrics Comparison (Lower is Better)",
                        yaxis_title="Geodesic Error (metres)",
                        barmode="group",
                        template="plotly_white",
                        height=380,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_bar, width="stretch")

                with ch2:
                    raw_s = np.sort(comp_res["raw_errors"])
                    rf_s = np.sort(comp_res["rf_errors"])
                    xgb_s = np.sort(comp_res["xgb_errors"])
                    pct = 100.0 * np.arange(len(raw_s)) / float(len(raw_s))
                    fig_cdf_cmp = go.Figure()
                    fig_cdf_cmp.add_trace(go.Scatter(
                        x=raw_s, y=pct, mode="lines", name="Raw GPS",
                        line=dict(color="#EF4444", width=2, dash="dash"),
                    ))
                    fig_cdf_cmp.add_trace(go.Scatter(
                        x=rf_s, y=pct, mode="lines", name="Random Forest",
                        line=dict(color="#10B981", width=2.5),
                    ))
                    fig_cdf_cmp.add_trace(go.Scatter(
                        x=xgb_s, y=pct, mode="lines", name="XGBoost",
                        line=dict(color="#2563EB", width=2.5),
                    ))
                    fig_cdf_cmp.update_layout(
                        title="Cumulative Error Distribution (CDF)",
                        xaxis_title="Geodesic Error (metres)",
                        yaxis_title="Percent of Observations (%)",
                        xaxis=dict(range=[0, min(10.0, float(np.percentile(raw_s, 99)))]),
                        template="plotly_white",
                        height=380,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_cdf_cmp, width="stretch")

                if not comp_res["per_trip_table"].empty and comp_res["n_trajectories"] > 1:
                    st.markdown("##### Per-Trajectory RMSE Breakdown")
                    st.dataframe(comp_res["per_trip_table"], width="stretch", hide_index=True)

    # Tab 3: Analytics & Error Diagnostics (Both Models Side-by-Side)
    with tab_analytics:
        st.subheader("Positioning Error Analysis (Raw vs Random Forest vs XGBoost)")
        if has_ground_truth and eval_metrics is not None:
            c1, c2 = st.columns([3, 2])
            
            with c1:
                # Error vs Epochs Time Series
                fig_err = go.Figure()
                fig_err.add_trace(go.Scatter(
                    y=feat_df["error_raw_m"],
                    mode="lines",
                    name="Raw GPS Error (m)",
                    line=dict(color="#EF4444", width=1.5),
                ))
                fig_err.add_trace(go.Scatter(
                    y=feat_df["error_rf_m"],
                    mode="lines",
                    name="Random Forest Error (m)",
                    line=dict(color="#8B5CF6", width=2),
                ))
                fig_err.add_trace(go.Scatter(
                    y=feat_df["error_xgb_m"],
                    mode="lines",
                    name="XGBoost Error (m)",
                    line=dict(color="#2563EB", width=2),
                ))
                fig_err.update_layout(
                    title="Positioning Error per Epoch (Metres)",
                    xaxis_title="Trajectory Epoch (Seconds)",
                    yaxis_title="Geodesic Error (Metres)",
                    template="plotly_white",
                    height=380,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_err, width="stretch")
                
            with c2:
                # Cumulative Distribution Function (CDF)
                raw_sorted = np.sort(feat_df["error_raw_m"].to_numpy(dtype=float))
                rf_sorted = np.sort(feat_df["error_rf_m"].to_numpy(dtype=float))
                xgb_sorted = np.sort(feat_df["error_xgb_m"].to_numpy(dtype=float))
                p = 100.0 * np.arange(len(raw_sorted)) / float(len(raw_sorted))
                
                fig_cdf = go.Figure()
                fig_cdf.add_trace(go.Scatter(
                    x=raw_sorted, y=p,
                    mode="lines", name="Raw GPS CDF",
                    line=dict(color="#EF4444", width=2, dash="dash")
                ))
                fig_cdf.add_trace(go.Scatter(
                    x=rf_sorted, y=p,
                    mode="lines", name="Random Forest CDF",
                    line=dict(color="#8B5CF6", width=2.5)
                ))
                fig_cdf.add_trace(go.Scatter(
                    x=xgb_sorted, y=p,
                    mode="lines", name="XGBoost CDF",
                    line=dict(color="#2563EB", width=2.5)
                ))
                fig_cdf.update_layout(
                    title="Error Cumulative Distribution (CDF)",
                    xaxis_title="Error Threshold (Metres)",
                    yaxis_title="Percent of Points (%)",
                    template="plotly_white",
                    height=380,
                    xaxis=dict(range=[0, min(12.0, float(np.percentile(raw_sorted, 99)))]),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_cdf, width="stretch")
                
            # Feature Importances for Both Models
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("##### Random Forest Feature Importances")
                rf_imp = rf_model.get_feature_importance().head(8)
                fig_rf_imp = px.bar(
                    rf_imp, x="Mean_Importance", y="Feature", orientation="h",
                    color="Mean_Importance", color_continuous_scale="Purples",
                )
                fig_rf_imp.update_layout(
                    height=260, margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False, coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_rf_imp, width="stretch")
                
            with c4:
                st.markdown("##### XGBoost Feature Importances")
                xgb_imp = xgb_model.get_feature_importance().head(8)
                fig_xgb_imp = px.bar(
                    xgb_imp, x="Mean_Importance", y="Feature", orientation="h",
                    color="Mean_Importance", color_continuous_scale="Blues",
                )
                fig_xgb_imp.update_layout(
                    height=260, margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False, coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_xgb_imp, width="stretch")
        else:
            st.info("Ground truth reference coordinates not present for this dataset. Analytics require ground truth.")

    # Tab 3: Real-Time Stream Simulator
    with tab_stream:
        st.subheader("Real-Time Streaming Simulator")
        st.markdown(
            "Simulates live GPS measurements streamed point-by-point into `RealTimeGpsCorrector`. "
            "Verifies causal state tracking, single-point inference latency, and live correction."
        )
        
        sim_cols = st.columns([1, 1, 2])
        max_sim = min(300, len(feat_df))
        with sim_cols[0]:
            if max_sim <= 2:
                sim_points = max_sim
                st.info(f"Simulation Length: {sim_points} points")
            else:
                min_sim = min(20, max_sim - 1)
                default_sim = min(60, max_sim)
                sim_points = st.slider("Simulation Length (Points)", min_sim, max_sim, default_sim)
        with sim_cols[1]:
            sim_delay = st.slider("Playback Speed (Interval ms)", 10, 200, 30)
            
        start_btn = st.button("▶ Start Real-Time Simulation", type="primary")
        
        if start_btn:
            rf_corrector = RealTimeGpsCorrector(model_type="random_forest", max_correction_meters=max_bound)
            xgb_corrector = RealTimeGpsCorrector(model_type="xgboost", max_correction_meters=max_bound)
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            metric_cols = st.columns(4)
            m1 = metric_cols[0].empty()
            m2 = metric_cols[1].empty()
            m3 = metric_cols[2].empty()
            m4 = metric_cols[3].empty()
            
            sim_subset = feat_df.iloc[:sim_points].reset_index(drop=True)
            latencies = []
            
            for step_idx, (_, row) in enumerate(sim_subset.iterrows()):
                t_ms = int(row["UnixTimeMillis"])
                lat = float(row["LatitudeDegrees_obs"])
                lon = float(row["LongitudeDegrees_obs"])
                
                res_rf = rf_corrector.feed(lat, lon, t_ms)
                res_xgb = xgb_corrector.feed(lat, lon, t_ms)
                latencies.append((res_rf.latency_ms + res_xgb.latency_ms) / 2.0)
                
                # Update live meters
                progress_bar.progress((step_idx + 1) / sim_points)
                m1.metric("Observed Lat/Lon", f"{res_rf.lat_original:.5f}, {res_rf.lon_original:.5f}")
                m2.metric("Random Forest Lat/Lon", f"{res_rf.lat_corrected:.5f}, {res_rf.lon_corrected:.5f}", f"{res_rf.correction_magnitude_m:.2f} m shift")
                m3.metric("XGBoost Lat/Lon", f"{res_xgb.lat_corrected:.5f}, {res_xgb.lon_corrected:.5f}", f"{res_xgb.correction_magnitude_m:.2f} m shift")
                m4.metric("Avg Latency", f"{latencies[-1]:.2f} ms")
                
                time.sleep(sim_delay / 1000.0)
                
            status_placeholder.success(
                f"✅ Real-Time Dual-Model Simulation Finished! Processed {sim_points} points. "
                f"Avg Latency: {np.mean(latencies):.2f} ms/point."
            )

    # Tab 5: Single Coordinate Manual Predictor (Both Models Simultaneously)
    with tab_single:
        st.subheader("Manual Coordinate Error Predictor (Simultaneous RF & XGBoost)")
        st.markdown("Passes identical GPS & kinematic inputs to both models independently.")
        
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            input_lat = st.number_input("Observed Latitude (°)", value=37.422050, format="%.6f")
            input_lon = st.number_input("Observed Longitude (°)", value=-122.084095, format="%.6f")
            input_speed = st.number_input("Vehicle Speed (m/s)", value=12.5, min_value=0.0, max_value=60.0)
        with p_col2:
            input_heading = st.number_input("Heading / Course (°)", value=45.0, min_value=0.0, max_value=360.0)
            input_turn = st.number_input("Turn Rate (°/s)", value=2.0, min_value=-45.0, max_value=45.0)
            input_res_x = st.number_input("Multipath Residual X (m)", value=1.2)
            input_res_y = st.number_input("Multipath Residual Y (m)", value=-1.8)
            
        if st.button("Predict Correction Offsets (Both Models)"):
            heading_rad = np.radians(input_heading)
            turn_rad = np.radians(input_turn)
            dx = input_speed * np.sin(heading_rad)
            dy = input_speed * np.cos(heading_rad)
            
            single_feat = pd.DataFrame([{
                "speed": input_speed,
                "accel": 0.0,
                "heading": heading_rad,
                "turn_rate": turn_rad,
                "dx": dx,
                "dy": dy,
                "dx_lag1": dx,
                "dy_lag1": dy,
                "dx_lag2": dx,
                "dy_lag2": dy,
                "res_x": input_res_x,
                "res_y": input_res_y,
                "speed_roll_mean_3": input_speed,
                "speed_roll_std_3": 0.5,
                "turn_roll_std_3": abs(turn_rad) * 0.5,
            }])
            
            pred_N_rf, pred_E_rf = rf_model.predict_errors(single_feat)
            e_N_rf = float(np.clip(pred_N_rf[0], -max_bound, max_bound))
            e_E_rf = float(np.clip(pred_E_rf[0], -max_bound, max_bound))

            pred_N_xgb, pred_E_xgb = xgb_model.predict_errors(single_feat)
            e_N_xgb = float(np.clip(pred_N_xgb[0], -max_bound, max_bound))
            e_E_xgb = float(np.clip(pred_E_xgb[0], -max_bound, max_bound))
            
            from src.feature_engineering import apply_corrections
            c_lat_rf, c_lon_rf = apply_corrections(
                np.array([input_lat]), np.array([input_lon]), np.array([e_N_rf]), np.array([e_E_rf])
            )
            c_lat_xgb, c_lon_xgb = apply_corrections(
                np.array([input_lat]), np.array([input_lon]), np.array([e_N_xgb]), np.array([e_E_xgb])
            )
            
            st.success("Independent Corrections Computed for Both Models!")
            rf_col, xgb_col = st.columns(2)
            with rf_col:
                st.markdown("##### Random Forest Output")
                st.metric("RF Corrected Latitude", f"{c_lat_rf[0]:.6f}", f"{c_lat_rf[0] - input_lat:+.6f}°")
                st.metric("RF Corrected Longitude", f"{c_lon_rf[0]:.6f}", f"{c_lon_rf[0] - input_lon:+.6f}°")
                st.metric("RF Shift Magnitude", f"{np.sqrt(e_N_rf**2 + e_E_rf**2):.2f} m")
            with xgb_col:
                st.markdown("##### XGBoost Output")
                st.metric("XGB Corrected Latitude", f"{c_lat_xgb[0]:.6f}", f"{c_lat_xgb[0] - input_lat:+.6f}°")
                st.metric("XGB Corrected Longitude", f"{c_lon_xgb[0]:.6f}", f"{c_lon_xgb[0] - input_lon:+.6f}°")
                st.metric("XGB Shift Magnitude", f"{np.sqrt(e_N_xgb**2 + e_E_xgb**2):.2f} m")

    # Tab 6: Data & Export
    with tab_data:
        st.subheader("Data Inspector & Export (Both Models)")
        export_cols = [
            "UnixTimeMillis", "LatitudeDegrees_obs", "LongitudeDegrees_obs",
            "lat_corr_rf", "lon_corr_rf", "lat_corr_xgb", "lon_corr_xgb",
            "speed", "heading"
        ]
        if has_ground_truth:
            export_cols += ["LatitudeDegrees_gt", "LongitudeDegrees_gt", "error_raw_m", "error_rf_m", "error_xgb_m"]
            
        export_df = feat_df[[c for c in export_cols if c in feat_df.columns]].copy()
        st.dataframe(export_df.head(100), width="stretch")
        
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Corrected Trajectory CSV",
            data=csv_buffer.getvalue(),
            file_name=f"corrected_trajectory_{int(time.time())}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
