"""
LEO Satellite Onboard AI Processing & Intelligent Data Transmission
Streamlit Dashboard

Sections:
  1. Sidebar – orbit controls, bandwidth, simulation speed
  2. 🌍 Globe – interactive satellite orbit visualisation [SIMULATED]
  3. 🛰 Onboard Processing – runs actual Eco_Act analysis
  4. 📡 Transmission Simulation – raw vs JSON comparison [SIMULATED]
  5. 📊 Ground-Station Dashboard – insights, payload, charts
"""

import sys
import os
import time
import json
import glob
from pathlib import Path

import streamlit as st
import numpy as np

# ── Path setup (make packages importable) ─────────────────────────────────────
_THIS_DIR   = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent   # Mach-A-Thon/
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(_THIS_DIR))

# ── Local imports ─────────────────────────────────────────────────────────────
from src.orbit import Satellite, OrbitalParams, GROUND_STATION
from src.communication import compare_transmission, build_json_payload, transmission_time_seconds
from src.satellite_viz import (
    build_globe_figure,
    build_size_comparison_chart,
)
from src.transmission_queue import TransmissionQueue
from onboard_processor.processor import run as run_processor
from onboard_processor.application_runner import APPLICATIONS, run as run_application

# ── Eco_Act dataset path ───────────────────────────────────────────────────────
ECO_ACT_DATASET = _PARENT_DIR / "Eco_Act" / "Dataset" / "monthly_10yr"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LEO Satellite AI Dashboard",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body, .stApp { background-color: #0d1117; color: #c9d1d9; }
  .metric-card {
    background: linear-gradient(135deg, #161b22, #1f2937);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .metric-title { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: #58a6ff; }
  .metric-delta { font-size: 0.8rem; color: #3fb950; }
  .simulated-badge {
    background-color: #21262d;
    border: 1px solid #f78166;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.7rem;
    color: #f78166;
    font-weight: 600;
  }
  .actual-badge {
    background-color: #21262d;
    border: 1px solid #3fb950;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.7rem;
    color: #3fb950;
    font-weight: 600;
  }
  h1,h2,h3 { color: #58a6ff; }
  .stSidebar { background-color: #161b22; }
  .stButton>button {
    background-color: #21262d;
    color: #58a6ff;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-weight: 600;
  }
  .stButton>button:hover { background-color: #1f6feb; color: white; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "sim_running":    False,
        "sim_elapsed_s":  0.0,
        "last_tick_time": None,
        "processed":      False,
        "metrics":        None,
        "comm_stats":     None,
        "image_path":     None,
        "sat":            None,
        "bandwidth_mbps": 2.0,
        "sim_speed":      1.0,
        "min_elevation_deg": 5.0,
        "analysis_application": "Eco_Act",
        "active_application": "Eco_Act",
        "urbanization_result": None,
        "application_result": None,
        "application_payload": None,
        "application_raw_bytes": None,
        "tx_queue": TransmissionQueue(),
        "queue_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_size(b: int) -> str:
    if b > 1_000_000:
        return f"{b/1_000_000:.2f} MB"
    if b > 1024:
        return f"{b/1024:.2f} KB"
    return f"{b} B"


def _metric_card(title: str, value: str, delta: str = "", badge: str = ""):
    badge_html = f'<span class="{"simulated-badge" if badge=="SIM" else "actual-badge"}">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-title">{title} {badge_html}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-delta">'+delta+'</div>' if delta else ''}
    </div>""", unsafe_allow_html=True)


def _get_dataset_images() -> list[str]:
    """Return sorted list of VIIRS image paths from Eco_Act dataset."""
    patterns = [
        str(ECO_ACT_DATASET / "*.png"),
        str(ECO_ACT_DATASET / "*.jpg"),
        str(ECO_ACT_DATASET / "*.tif"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰 LEO Satellite Controls")

    # Orbit parameters display
    st.markdown("### Orbital Parameters")
    params = OrbitalParams(altitude_km=500.0, inclination_deg=97.4)
    sat = Satellite(params)
    if st.session_state["sat"] is None:
        st.session_state["sat"] = sat

    summary = sat.summary()
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Altitude | **{summary['altitude_km']} km** |
    | Inclination | **{summary['inclination_deg']}°** |
    | Period | **{summary['orbital_period_minutes']:.1f} min** |
    | Velocity | **{summary['velocity_km_per_s']:.4f} km/s** |
    | Shape | Circular |
    """)
    st.caption("⚠ Period computed from Kepler's 3rd law: T = 2π√(a³/μ)")

    st.divider()
    # Simulation controls
    st.markdown("### ⏱ Simulation Controls")
    sim_speed = st.slider("Simulation Speed (×)", 0.1, 100.0, 1.0, 0.1)
    st.session_state["sim_speed"] = sim_speed

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Start"):
            st.session_state["sim_running"] = True
            st.session_state["last_tick_time"] = time.time()
    with col2:
        if st.button("⏸ Pause"):
            st.session_state["sim_running"] = False
    with col3:
        if st.button("⟳ Reset"):
            st.session_state["sim_running"] = False
            st.session_state["sim_elapsed_s"] = 0.0
            st.session_state["last_tick_time"] = None
            st.session_state["processed"] = False
            st.session_state["metrics"] = None
            st.session_state["comm_stats"] = None
            st.session_state["tx_queue"] = TransmissionQueue()
            st.session_state["queue_result"] = None
            st.session_state["application_result"] = None
            st.session_state["application_payload"] = None
            st.session_state["application_raw_bytes"] = None

    st.divider()
    # Bandwidth setting
    st.markdown("### 📡 Communication")
    bw = st.slider("Bandwidth (Mbps)", 0.5, 20.0, 2.0, 0.5)
    st.session_state["bandwidth_mbps"] = bw
    st.caption("[SIMULATED] Communication bandwidth")

    min_el = st.slider("Minimum elevation mask (°)", 0.0, 30.0, 5.0, 1.0)
    st.session_state["min_elevation_deg"] = min_el

    st.divider()
    # Ground station info
    gs = GROUND_STATION
    st.markdown("### 📡 Ground Station")
    st.markdown(f"""
    **{gs['name']}**  
    {gs['city']}  
    Lat: {gs['lat']}° · Lon: {gs['lon']}°
    """)


# ── Advance simulation clock ──────────────────────────────────────────────────
if st.session_state["sim_running"] and st.session_state["last_tick_time"]:
    now = time.time()
    real_dt = now - st.session_state["last_tick_time"]
    st.session_state["sim_elapsed_s"] += real_dt * st.session_state["sim_speed"]
    st.session_state["last_tick_time"] = now

elapsed_s = st.session_state["sim_elapsed_s"]
sat_obj   = st.session_state["sat"]
sat_pos   = sat_obj.position_at(elapsed_s)
visibility = sat_obj.visibility_at(elapsed_s, gs["lat"], gs["lon"], st.session_state["min_elevation_deg"])
in_view   = visibility["visible"]
next_visibility = sat_obj.next_visibility_seconds(
    elapsed_s, gs["lat"], gs["lon"], st.session_state["min_elevation_deg"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
# 🛰 LEO Satellite Onboard AI Processing & Intelligent Data Transmission
<span class="simulated-badge">SIM</span> Orbit & Communication are SIMULATED &nbsp;&nbsp;
<span class="actual-badge">ACTUAL</span> Image analysis uses real Eco_Act computations
""", unsafe_allow_html=True)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 – Globe
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🌍 Satellite Orbit [SIMULATED]")

col_globe, col_info = st.columns([3, 1])

with col_globe:
    ground_track = sat_obj.ground_track(n_points=360)
    fig_globe = build_globe_figure(
        ground_track=ground_track,
        sat_pos=sat_pos,
        gs=gs,
        sat_in_view=in_view,
        show_comm_link=True,
    )
    st.plotly_chart(fig_globe, use_container_width=True)

with col_info:
    st.markdown("### Satellite Status")
    _metric_card("Latitude",  f"{sat_pos['lat']:.3f}°", badge="SIM")
    _metric_card("Longitude", f"{sat_pos['lon']:.3f}°", badge="SIM")
    _metric_card("Altitude",  f"{sat_pos['altitude_km']} km", badge="SIM")
    _metric_card("Orbit Fraction", f"{sat_pos['orbit_fraction']*100:.1f}%", badge="SIM")
    _metric_card("Elevation", f"{visibility['elevation_deg']:.2f}°",
                 f"Azimuth {visibility['azimuth_deg']:.1f}°", badge="SIM")
    _metric_card("Slant Range", f"{visibility['distance_km']:.2f} km",
                 f"Mask {visibility['min_elevation_deg']:.0f}°", badge="SIM")
    link_color = "🟢" if in_view else "🔴"
    st.markdown(f"**Ground Link:** {link_color} {'IN VIEW' if in_view else 'OUT OF VIEW'}")
    st.caption("[SIMULATED]")
    if next_visibility is None:
        st.caption("No next contact found in the four-orbit search horizon.")
    elif not in_view:
        st.caption(f"Next visibility opportunity in approximately {next_visibility:.0f} s.")

    st.markdown("### Sim Clock")
    T = sat_obj.params.period_seconds
    orbit_num = int(elapsed_s // T) + 1
    st.markdown(f"**Elapsed:** {elapsed_s:.1f} s")
    st.markdown(f"**Orbits completed:** {orbit_num - 1}")

# Auto-rerun while simulation is running (refresh every 0.5 s)
if st.session_state["sim_running"]:
    time.sleep(0.5)
    st.rerun()

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 – Image acquisition & onboard processing
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🛰 Onboard AI Processing")
analysis_application = st.selectbox(
    "Onboard analysis application", list(APPLICATIONS), key="analysis_application",
    help="Each application uses its own supported inputs and existing inference pipeline.",
)
if st.session_state["active_application"] != analysis_application:
    st.session_state["active_application"] = analysis_application
    st.session_state["processed"] = False
    st.session_state["metrics"] = None
    st.session_state["comm_stats"] = None
    st.session_state["application_result"] = None
    st.session_state["application_payload"] = None
priority = st.selectbox("Transmission priority", ["HIGH", "MEDIUM", "LOW"], key="priority_after_run")

if analysis_application == "Eco_Act":
    col_img_sel, col_process = st.columns([2, 1])
    with col_img_sel:
        st.markdown("### VIIRS Image Acquisition")
        dataset_images = _get_dataset_images()
        img_source = st.radio("Image source", ["Select from Eco_Act dataset", "Upload a VIIRS image"], horizontal=True)
        selected_image_path = None
        if img_source == "Select from Eco_Act dataset":
            if dataset_images:
                basenames = [os.path.basename(p) for p in dataset_images]
                choice = st.selectbox("Select VIIRS image", basenames, index=len(basenames)-1)
                selected_image_path = dataset_images[basenames.index(choice)]
            else:
                st.warning(f"No images found in: {ECO_ACT_DATASET}")
        else:
            uploaded = st.file_uploader("Upload VIIRS image", type=["png", "jpg", "jpeg", "tif"])
            if uploaded:
                tmp_path = _THIS_DIR / "tmp_upload.png"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                selected_image_path = str(tmp_path)
        if selected_image_path and os.path.exists(selected_image_path):
            from PIL import Image as PILImage
            img_pil = PILImage.open(selected_image_path).convert("RGB")
            st.image(img_pil, caption=f"📷 {os.path.basename(selected_image_path)}", use_column_width=True)
            raw_size_b = os.path.getsize(selected_image_path)
            st.caption(f"File size: **{_fmt_size(raw_size_b)}** · Dimensions: {img_pil.size[0]}×{img_pil.size[1]} px")
            st.session_state["image_path"] = selected_image_path

    with col_process:
        st.markdown("### Processing Controls")
        if st.session_state.get("image_path") and os.path.exists(st.session_state["image_path"]):
            if st.button("🚀 Run Onboard Analysis", use_container_width=True):
                ip = st.session_state["image_path"]
                with st.spinner("⚙ Onboard processor executing Eco_Act…"):
                    try:
                        t0 = time.perf_counter()
                        metrics = run_processor(ip)
                        elapsed = time.perf_counter() - t0
                        metrics["processing_time_seconds"] = round(elapsed, 4)
                        st.session_state["metrics"] = metrics
                        st.session_state["processed"] = True
                        st.session_state["application_result"] = None
                        st.session_state["application_payload"] = None
                        st.session_state["application_raw_bytes"] = None
                        bw = st.session_state["bandwidth_mbps"]
                        st.session_state["comm_stats"] = compare_transmission(ip, metrics, bandwidth_mbps=bw)
                        st.session_state["tx_queue"].enqueue(
                            build_json_payload(metrics), priority, label=os.path.basename(ip),
                            metadata={"application": "Eco_Act"},
                        )
                        st.success(f"✅ Processing complete in {elapsed:.3f} s; payload queued as {priority}")
                    except Exception as ex:
                        st.error(f"❌ Eco_Act processor error: {ex}")
        else:
            st.info("Select or upload a VIIRS image to enable processing.")
        if st.session_state.get("processed") and st.session_state.get("metrics"):
            m = st.session_state["metrics"]
            st.markdown("#### Processor Output")
            st.markdown(f"⏱ **Processing time:** `{m['processing_time_seconds']} s`")
            st.markdown("💡 **Features extracted:** brightness, lit area, industrial activity, economic index, GDDP, opportunity zones")

elif analysis_application == "UrbanPulse":
    from urbanpulse_adapter import get_preset_hubs
    PRESET_HUBS = get_preset_hubs()
    hub_names = list(PRESET_HUBS)
    hub_name = st.selectbox("UrbanPulse feature-layer location", hub_names,
                            index=hub_names.index("Tumakuru / Tumkur (Vasanthanarasapura – NH-48 Industrial)"))
    hub = PRESET_HUBS[hub_name]
    st.caption(f"Existing UrbanPulse pipeline uses its aligned 48×48 Sentinel, night-light and OSM feature layers at ({hub['lat']}, {hub['lon']}).")
    if st.button("🚀 Run UrbanPulse Inference", use_container_width=True):
        try:
            with st.spinner("Fetching compatible layers and running UrbanPulse classifier and forecaster…"):
                result = run_application("UrbanPulse", lat=hub["lat"], lon=hub["lon"])
                payload = build_json_payload(result)
                st.session_state["application_result"] = result
                st.session_state["application_payload"] = payload
                st.session_state["application_raw_bytes"] = None
                st.session_state["processed"] = False
                st.session_state["metrics"] = None
                st.session_state["comm_stats"] = None
                st.session_state["tx_queue"].enqueue(
                    payload, priority, label=f"UrbanPulse {hub_name}",
                    metadata={"application": "UrbanPulse", "location": hub_name},
                )
            st.success(f"UrbanPulse completed and queued as {priority} ({len(payload):,} bytes).")
        except Exception as ex:
            st.error(f"UrbanPulse inference failed: {ex}")

elif analysis_application == "DisasterRisk":
    from disasterrisk_adapter import list_scenarios
    scenarios = list_scenarios()
    scenario_ids = list(scenarios)
    scenario_id = st.selectbox("DisasterRisk event scenario", scenario_ids,
                               format_func=lambda key: scenarios[key]["title"])
    st.caption("Runs the existing ClaimPulse connector → hazard → exposure → loss → decision pipeline. Preset scenarios are identified as demo data in the payload.")
    if st.button("🚀 Run DisasterRisk Inference", use_container_width=True):
        try:
            with st.spinner("Running DisasterRisk scenario decision pipeline…"):
                result = run_application("DisasterRisk", preset_id=scenario_id)
                payload = build_json_payload(result)
                st.session_state["application_result"] = result
                st.session_state["application_payload"] = payload
                st.session_state["application_raw_bytes"] = None
                st.session_state["processed"] = False
                st.session_state["metrics"] = None
                st.session_state["comm_stats"] = None
                st.session_state["tx_queue"].enqueue(
                    payload, priority, label=f"DisasterRisk {result['event']['title']}",
                    metadata={"application": "DisasterRisk", "scenario": scenario_id},
                )
            st.success(f"DisasterRisk completed and queued as {priority} ({len(payload):,} bytes).")
        except Exception as ex:
            st.error(f"DisasterRisk inference failed: {ex}")

if st.session_state.get("application_result"):
    st.markdown("#### Application inference output")
    st.json(st.session_state["application_result"])

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 – Analysis Results (shown after processing)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("processed") and st.session_state.get("metrics"):
    m = st.session_state["metrics"]
    st.markdown("## 📊 Extracted Economic Insights [ACTUAL]")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        _metric_card("Mean Radiance",
                     f"{m['brightness']['mean']:.2f} DN",
                     f"Max: {m['brightness']['max']:.1f} DN", badge="ACTUAL")
    with r2:
        _metric_card("Lit Area",
                     f"{m['lit_area_pct']:.2f}%",
                     f"{m['lit_pixel_count']:,} lit pixels", badge="ACTUAL")
    with r3:
        _metric_card("Industrial Activity",
                     m["industrial_activity_trend"],
                     f"Ratio: {m['industrial_pixel_ratio']*100:.3f}%", badge="ACTUAL")
    with r4:
        _metric_card("Economic Index",
                     f"{m['economic_activity_index']:.1f} / 100",
                     badge="ACTUAL")

    r5, r6, r7 = st.columns(3)
    gddp = m["gddp"]
    with r5:
        _metric_card("Est. GDDP",
                     f"₹{gddp['predicted_gddp_crores']:,.0f} Cr",
                     f"CI: [{gddp['confidence_low_crores']:,.0f} – {gddp['confidence_high_crores']:,.0f}]",
                     badge="ACTUAL")
    with r6:
        _metric_card("Sum of Lights (SOL)",
                     f"{m['sum_of_lights']:,.0f}",
                     f"SOL ratio: {gddp['sol_ratio']:.4f}", badge="ACTUAL")
    with r7:
        _metric_card("Processing Time",
                     f"{m['processing_time_seconds']} s",
                     "Wall-clock onboard processing", badge="ACTUAL")

    # Opportunity zones
    st.markdown("### 🏙 Opportunity Zones [ACTUAL – derived from Eco_Act analysis]")
    zones = m.get("opportunity_zones", [])
    if zones:
        import pandas as pd
        df_zones = pd.DataFrame([
            {
                "Zone": z["name"],
                "Type": z["type"],
                "Mean Radiance": z["mean_radiance"],
                "Growth % (1yr forecast)": z["growth_pct"],
                "Current Flux": z["current_flux"],
                "Status": z["status"],
                "Recommended Action": z["recommended_action"],
            }
            for z in zones[:8]   # top 8
        ])
        st.dataframe(df_zones, use_container_width=True, hide_index=True)
        st.caption("Growth forecast uses Eco_Act Gaussian diffusion model (1-year horizon). **Forecast – not historical.**")
    else:
        st.info("No opportunity zones extracted.")

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 – Transmission Simulation
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("comm_stats"):
    cs = st.session_state["comm_stats"]
    bw_now = st.session_state["bandwidth_mbps"]

    # Re-compute if bandwidth changed
    if abs(cs["bandwidth_mbps"] - bw_now) > 0.001 and st.session_state.get("metrics"):
        cs = compare_transmission(
            st.session_state["image_path"],
            st.session_state["metrics"],
            bandwidth_mbps=bw_now,
        )
        st.session_state["comm_stats"] = cs

    st.markdown("## 📡 Transmission Simulation [SIMULATED]")
    st.caption("⚠ Communication scenarios are simulated. File sizes are ACTUAL; transmission timing is calculated from actual sizes and configured bandwidth.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Scenario A – Raw Image Transmission")
        _metric_card("Image Size",
                     _fmt_size(cs["raw_image_size_bytes"]),
                     f"{cs['raw_image_size_bytes']:,} bytes", badge="ACTUAL")
        _metric_card("Transmission Time",
                     f"{cs['raw_transmission_time_s']:.4f} s",
                     f"@ {bw_now} Mbps", badge="SIM")

        if st.button("▶ Simulate Raw Transmission", use_container_width=True):
            prog = st.progress(0, text="Transmitting raw image…")
            for i in range(0, 101, 5):
                prog.progress(i, text=f"[SIMULATED] Sending raw image… {i}%")
                time.sleep(0.03)
            st.success(f"✅ [SIMULATED] Raw image received at ground station ({_fmt_size(cs['raw_image_size_bytes'])})")

    with col_b:
        st.markdown("### Scenario B – Intelligent JSON Transmission")
        _metric_card("JSON Payload Size",
                     _fmt_size(cs["json_payload_size_bytes"]),
                     f"{cs['json_payload_size_bytes']:,} bytes", badge="ACTUAL")
        _metric_card("Transmission Time",
                     f"{cs['json_transmission_time_s']:.6f} s",
                     f"@ {bw_now} Mbps", badge="SIM")

        if st.button("▶ Simulate Intelligent Transmission", use_container_width=True):
            prog = st.progress(0, text="Transmitting JSON insights…")
            for i in range(0, 101, 10):
                prog.progress(i, text=f"[SIMULATED] Sending JSON payload… {i}%")
                time.sleep(0.02)
            st.success(f"✅ [SIMULATED] JSON insights received at ground station ({_fmt_size(cs['json_payload_size_bytes'])})")

    # Key statistics
    st.markdown("### 📉 Transmission Statistics")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        _metric_card("Data Reduction",
                     f"{cs['data_reduction_pct']:.2f}%",
                     f"{cs['size_reduction_ratio']:.0f}× smaller", badge="ACTUAL")
    with s2:
        _metric_card("Time Saved",
                     f"{cs['time_saved_s']:.4f} s",
                     f"@ {bw_now} Mbps", badge="SIM")
    with s3:
        _metric_card("Raw Image",
                     _fmt_size(cs["raw_image_size_bytes"]),
                     badge="ACTUAL")
    with s4:
        _metric_card("JSON Payload",
                     _fmt_size(cs["json_payload_size_bytes"]),
                     badge="ACTUAL")

    st.markdown("---")

    # Comparison chart
    st.markdown("### 📊 Raw Image vs JSON Insight Comparison")
    fig_cmp = build_size_comparison_chart(cs)
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")

    # Ground station received payload
    st.markdown("## 📡 Ground Station: Received JSON Payload")
    col_json, col_meta = st.columns([3, 1])
    with col_json:
        st.markdown("#### Received Insight Payload [ACTUAL content]")
        st.code(cs["json_payload_str"], language="json")
    with col_meta:
        st.markdown("#### Payload Metadata")
        _metric_card("Payload Size",   _fmt_size(cs["json_payload_size_bytes"]), badge="ACTUAL")
        _metric_card("Bandwidth Used", f"{bw_now} Mbps", badge="SIM")
        _metric_card("Tx Time",        f"{cs['json_transmission_time_s']:.6f} s", badge="SIM")
        _metric_card("Reduction",      f"{cs['data_reduction_pct']:.2f}%", badge="ACTUAL")

    st.markdown("### Adaptive transmission queue")
    queue = st.session_state["tx_queue"]
    pending = queue.snapshot()
    if pending:
        import pandas as pd
        st.dataframe(pd.DataFrame(pending), use_container_width=True, hide_index=True)
        contact_seconds = st.number_input("Available contact budget (seconds)", 0.0, 600.0, 10.0, 1.0)
        if st.button("Dispatch queued observations", disabled=not in_view, use_container_width=True):
            st.session_state["queue_result"] = queue.dispatch(
                visible=in_view,
                bandwidth_mbps=bw_now,
                contact_seconds=contact_seconds,
            )
            st.rerun()
        if not in_view:
            st.warning("Queue is deferred: the satellite is below the configured elevation mask.")
    else:
        st.info("No queued observations. Run an application to enqueue its insight payload.")
    if st.session_state.get("queue_result"):
        result = st.session_state["queue_result"]
        st.write(f"Sent: {len(result['sent'])} · Deferred: {len(result['deferred'])} · Remaining budget: {result['remaining_bytes']:.0f} bytes")
        if result["sent"]:
            st.success("Ground station received: " + ", ".join(item.label for item in result["sent"]))
            for item in result["sent"]:
                st.markdown(f"**Received {item.metadata.get('application', 'application')} payload**")
                st.code(item.payload.decode("utf-8"), language="json")

    st.markdown("---")

    # Summary banner
    st.markdown(f"""
    <div style='background:linear-gradient(90deg,#0d2137,#1a3a52);border:1px solid #1f6feb;
                border-radius:12px;padding:20px;text-align:center;'>
      <h3 style='color:#58a6ff;margin:0'>🏆 Onboard AI Processing Result</h3>
      <p style='color:#c9d1d9;margin:8px 0'>
        Transmitting the compact JSON insight payload instead of the raw VIIRS image
        reduces data volume by <strong style='color:#3fb950'>{cs['data_reduction_pct']:.1f}%</strong>
        — a <strong style='color:#3fb950'>{cs['size_reduction_ratio']:.0f}×</strong> reduction —
        and saves <strong style='color:#3fb950'>{cs['time_saved_s']:.4f} s</strong> of link time
        at {bw_now} Mbps.
        <br><small style='color:#8b949e'>File sizes: ACTUAL &nbsp;|&nbsp; Transmission timing: SIMULATED</small>
      </p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.get("application_payload"):
    payload = st.session_state["application_payload"]
    bw_now = st.session_state["bandwidth_mbps"]
    tx_seconds = transmission_time_seconds(len(payload), bw_now)
    st.markdown("## 📡 Transmission & Tumkur Ground Station")
    st.caption("Application output is serialized as its own JSON insight payload and uses the shared LEO transmission queue.")
    col_payload, col_link = st.columns([3, 1])
    with col_payload:
        st.markdown("#### Onboard JSON insight payload")
        st.code(payload.decode("utf-8"), language="json")
    with col_link:
        _metric_card("Payload Size", _fmt_size(len(payload)), f"{len(payload):,} bytes", badge="ACTUAL")
        _metric_card("JSON Link Time", f"{tx_seconds:.6f} s", f"@ {bw_now} Mbps", badge="SIM")
        _metric_card("Ground Link", "IN VIEW" if in_view else "OUT OF VIEW", badge="SIM")

    st.markdown("### Adaptive transmission queue")
    queue = st.session_state["tx_queue"]
    pending = queue.snapshot()
    if pending:
        import pandas as pd
        st.dataframe(pd.DataFrame(pending), use_container_width=True, hide_index=True)
        contact_seconds = st.number_input("Available contact budget (seconds)", 0.0, 600.0, 10.0, 1.0,
                                          key="application_contact_budget")
        if st.button("Dispatch queued observations", disabled=not in_view, use_container_width=True,
                     key="dispatch_application_payloads"):
            st.session_state["queue_result"] = queue.dispatch(
                visible=in_view, bandwidth_mbps=bw_now, contact_seconds=contact_seconds,
            )
            st.rerun()
        if not in_view:
            st.warning("Queue is deferred: the satellite is below the configured elevation mask.")
    else:
        st.info("No queued observations.")
    if st.session_state.get("queue_result"):
        result = st.session_state["queue_result"]
        st.write(f"Sent: {len(result['sent'])} · Deferred: {len(result['deferred'])} · Remaining budget: {result['remaining_bytes']:.0f} bytes")
        for item in result["sent"]:
            st.success(f"Tumkur ground station received {item.label} ({item.priority})")
            st.code(item.payload.decode("utf-8"), language="json")

else:
    if not st.session_state.get("processed"):
        st.info("💡 Select an onboard application and its compatible input, then run the analysis to see results.")
