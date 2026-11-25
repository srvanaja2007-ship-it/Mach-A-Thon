"""
app.py — UrbanPulse AI | Aerospace Command Center Dashboard
Aeromania '26 Mach-A-Thon | Track A: Migration & Urbanization
Boeing-pitch ready: sleek, modern, executive-grade geospatial intelligence.
"""

from __future__ import annotations

import sys
import os
import json
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure the project directory is on the Python path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, MiniMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_pipeline import PRESET_HUBS, fetch_location_data, _cache_key, CACHE_DIR
from ml_engine import run_ml_pipeline, ZONE_LABELS, ZONE_COLORS, ZONE_SHORT
from planner_engine import run_planner, simulate_roi

log = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG — must be the very first Streamlit call
# ============================================================================
st.set_page_config(
    page_title="UrbanPulse AI | Orbit-to-Economy",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# GLOBAL CSS — Aerospace Command Center Theme
# ============================================================================
CUSTOM_CSS = """
<style>
  /* ---- Reset & base ---- */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0.6rem 1.2rem 1rem 1.2rem !important; }
  html, body, [class*="css"] {
      background-color: #070B14 !important;
      color: #E2E8F0 !important;
      font-family: 'Inter', 'Segoe UI', sans-serif;
  }

  /* ---- Sidebar ---- */
  section[data-testid="stSidebar"] {
      background: #0A1020 !important;
      border-right: 1px solid #1E293B !important;
  }
  section[data-testid="stSidebar"] .block-container {
      padding: 1rem !important;
  }

  /* ---- Header strip ---- */
  .header-strip {
      background: linear-gradient(135deg, #0F172A 0%, #0C1829 50%, #0F172A 100%);
      border: 1px solid #1E3A5F;
      border-radius: 12px;
      padding: 18px 24px 14px 24px;
      margin-bottom: 10px;
      box-shadow: 0 0 40px rgba(6,182,212,0.08);
  }
  .header-title {
      font-size: 1.75rem;
      font-weight: 800;
      background: linear-gradient(90deg, #06B6D4, #38BDF8, #06B6D4);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.5px;
  }
  .header-subtitle {
      font-size: 0.82rem;
      color: #64748B;
      margin-top: 2px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
  }
  .sensor-pill {
      display: inline-block;
      background: rgba(6,182,212,0.1);
      border: 1px solid #164E63;
      border-radius: 20px;
      padding: 2px 10px;
      font-size: 0.68rem;
      font-weight: 600;
      color: #38BDF8;
      letter-spacing: 0.8px;
      margin-right: 6px;
      margin-top: 8px;
  }
  .sensor-pill.active { border-color: #06B6D4; color: #06B6D4; }

  /* ---- Value banner ---- */
  .value-banner {
      background: linear-gradient(135deg, #1a0a00 0%, #1f1200 100%);
      border: 1px solid #92400E;
      border-left: 4px solid #F59E0B;
      border-radius: 8px;
      padding: 10px 16px;
      margin: 6px 0 10px 0;
      font-size: 0.82rem;
      color: #FCD34D;
  }
  .value-banner strong { color: #FBBF24; font-size: 0.88rem; }

  /* ---- KPI Cards ---- */
  .kpi-card {
      background: #0F172A;
      border: 1px solid #1E293B;
      border-radius: 12px;
      padding: 16px 18px;
      text-align: center;
      position: relative;
      overflow: hidden;
      transition: all 0.3s ease;
  }
  .kpi-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
  }
  .kpi-card.cyan::before  { background: linear-gradient(90deg, transparent, #06B6D4, transparent); }
  .kpi-card.red::before   { background: linear-gradient(90deg, transparent, #EF4444, transparent); }
  .kpi-card.amber::before { background: linear-gradient(90deg, transparent, #F59E0B, transparent); }
  .kpi-card.green::before { background: linear-gradient(90deg, transparent, #22C55E, transparent); }

  .kpi-value {
      font-size: 2.0rem;
      font-weight: 800;
      line-height: 1;
      margin: 6px 0;
  }
  .kpi-value.cyan  { color: #06B6D4; }
  .kpi-value.red   { color: #EF4444; }
  .kpi-value.amber { color: #F59E0B; }
  .kpi-value.green { color: #22C55E; }

  .kpi-label {
      font-size: 0.72rem;
      color: #64748B;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-weight: 600;
  }
  .kpi-delta {
      font-size: 0.75rem;
      margin-top: 4px;
  }

  /* ---- Section headers ---- */
  .section-head {
      font-size: 0.72rem;
      color: #38BDF8;
      text-transform: uppercase;
      letter-spacing: 2px;
      font-weight: 700;
      border-bottom: 1px solid #1E293B;
      padding-bottom: 4px;
      margin: 14px 0 8px 0;
  }

  /* ---- Sidebar controls ---- */
  .stSelectbox label, .stSlider label, .stTextInput label,
  .stButton label, [data-testid="stWidgetLabel"] {
      color: #94A3B8 !important;
      font-size: 0.78rem !important;
      font-weight: 600 !important;
      text-transform: uppercase;
      letter-spacing: 0.8px;
  }
  .stSelectbox > div > div {
      background: #0F172A !important;
      border: 1px solid #1E3A5F !important;
      border-radius: 8px !important;
      color: #E2E8F0 !important;
  }
  .stTextInput > div > div > input {
      background: #0F172A !important;
      border: 1px solid #1E3A5F !important;
      border-radius: 8px !important;
      color: #E2E8F0 !important;
  }

  /* ---- Buttons ---- */
  .stButton > button {
      background: linear-gradient(135deg, #0E7490, #0369A1) !important;
      color: white !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
      font-size: 0.82rem !important;
      letter-spacing: 0.5px !important;
      padding: 0.6rem 1.2rem !important;
      width: 100%;
      box-shadow: 0 0 20px rgba(6,182,212,0.3) !important;
      transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
      box-shadow: 0 0 30px rgba(6,182,212,0.6) !important;
      transform: translateY(-1px) !important;
  }

  /* ---- Tables ---- */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
  thead tr th {
      background: #0F172A !important;
      color: #38BDF8 !important;
      font-size: 0.72rem !important;
      text-transform: uppercase;
      letter-spacing: 1px;
  }

  /* ---- Tab styling ---- */
  .stTabs [data-baseweb="tab-list"] {
      background: #0A1020 !important;
      border-radius: 8px 8px 0 0;
      gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
      color: #64748B !important;
      font-size: 0.78rem !important;
      font-weight: 600 !important;
      letter-spacing: 0.5px;
      padding: 10px 18px !important;
  }
  .stTabs [aria-selected="true"] {
      background: #0F172A !important;
      color: #06B6D4 !important;
      border-bottom: 2px solid #06B6D4 !important;
  }

  /* ---- Folium map container ---- */
  iframe { border-radius: 10px; border: 1px solid #1E293B !important; }

  /* ---- Spinner / progress ---- */
  .stSpinner > div { border-top-color: #06B6D4 !important; }

  /* ---- Alert boxes ---- */
  .stAlert { border-radius: 8px !important; }

  /* ---- Plotly charts dark bg ---- */
  .js-plotly-plot { border-radius: 10px; }

  /* ---- Architecture card ---- */
  .arch-card {
      background: #0F172A;
      border: 1px solid #1E3A5F;
      border-radius: 10px;
      padding: 14px 18px;
      margin: 6px 0;
  }
  .arch-card .module-name {
      font-size: 0.88rem;
      font-weight: 700;
      color: #06B6D4;
      margin-bottom: 4px;
  }
  .arch-card .module-desc {
      font-size: 0.78rem;
      color: #94A3B8;
      line-height: 1.5;
  }
  .arch-arrow {
      text-align: center;
      color: #38BDF8;
      font-size: 1.2rem;
      margin: 2px 0;
  }

  /* ---- Status indicator ---- */
  .status-dot {
      display: inline-block;
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #22C55E;
      box-shadow: 0 0 6px #22C55E;
      margin-right: 5px;
  }
  .status-dot.warn { background: #F59E0B; box-shadow: 0 0 6px #F59E0B; }
</style>
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def _get_base_plan(lat: float, lon: float) -> dict:
    """Cached heavy satellite + ML + corridor routing pipeline."""
    data   = fetch_location_data(lat, lon)
    ml_res = run_ml_pipeline(data)
    return run_planner(ml_res, early_budget_cr=50.0)


def _run_full_pipeline(lat: float, lon: float, budget_cr: float) -> dict:
    """Instant ROI update on cached satellite/ML/corridor results."""
    base_plan = _get_base_plan(lat, lon)
    roi = simulate_roi(base_plan, base_plan["corridors"], early_budget_cr=budget_cr)
    return {**base_plan, "roi": roi}


def _fmt_cr(x: float) -> str:
    if abs(x) >= 1000:
        return f"₹{x/1000:.1f}K Cr"
    return f"₹{x:.1f} Cr"


def _fmt_m(x: float) -> str:
    return f"${x:.1f} M"


def _geocode_location(query: str) -> tuple[float, float] | None:
    """
    Try to parse 'lat,lon' directly, then fall back to Nominatim geocoder.
    """
    # Try direct lat,lon parse
    parts = query.replace(" ", "").split(",")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    # Nominatim geocode
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="urbanpulse_ai_v1")
        loc = geolocator.geocode(query + ", India", timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except Exception as e:
        st.warning(f"Geocoding failed: {e}")
    return None


# ============================================================================
# MAP BUILDER
# ============================================================================

ZONE_FOLIUM_COLOR = {
    0: "#06B6D4",   # Cyan — tech core
    1: "#22C55E",   # Green — legal suburb
    2: "#EF4444",   # Red — NLG alert
    3: "#84CC16",   # Lime — green buffer
}


def _build_map(plan: dict,
               show_zones: bool = True,
               show_nlg: bool = True,
               show_forecast: bool = True,
               show_corridors: bool = True) -> folium.Map:
    """Build the full interactive Folium command-center map."""
    lat, lon = plan["lat"], plan["lon"]
    n = plan["n_cells"]

    m = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        max_zoom=16,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )
    # Add satellite imagery toggle (capped at max_zoom=19)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite Imagery",
        max_zoom=19,
        overlay=False,
        control=True,
    ).add_to(m)

    # --- Zone grid overlay ---
    if show_zones:
        zone_layer = folium.FeatureGroup(name="4-Zone AI Classification", show=True)
        zone_preds = plan["zone_preds"]
        grid_lat   = plan["grid_lat"]
        grid_lon   = plan["grid_lon"]
        half_cell  = 0.00113   # ~125m in degrees for 250m cell

        # Sample every 2nd cell for performance (still 24×24 = 576 squares)
        step = 2
        for r in range(0, n, step):
            for c in range(0, n, step):
                z     = int(zone_preds[r, c])
                clat  = float(grid_lat[r, c])
                clon  = float(grid_lon[r, c])
                color = ZONE_FOLIUM_COLOR.get(z, "#64748B")
                alpha = 0.55 if z == 2 else 0.30
                bounds = [
                    [clat - half_cell, clon - half_cell],
                    [clat + half_cell, clon + half_cell],
                ]
                folium.Rectangle(
                    bounds=bounds,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=alpha,
                    weight=0,
                    popup=folium.Popup(
                        f"""<div style="background:#0F172A;color:#E2E8F0;padding:10px;
                                       border-radius:8px;font-family:monospace;font-size:11px;
                                       border:1px solid #1E3A5F;min-width:200px">
                          <b style="color:{color}">{ZONE_LABELS.get(z,'Unknown')}</b><br>
                          Lat: {clat:.5f} | Lon: {clon:.5f}<br>
                          ΔNDBI: {plan['delta_ndbi'][r,c]:.3f}<br>
                          Radar VV: {plan['vv_db'][r,c]:.1f} dB<br>
                          VIIRS: {plan['viirs_dnb'][r,c]:.1f} nW/cm²/sr<br>
                          Road dist: {plan['road_dist_m'][r,c]:.0f} m
                        </div>""",
                        max_width=280,
                    ),
                ).add_to(zone_layer)
        zone_layer.add_to(m)

    # --- Night-Light Gap alert zones (NLG — Zone 2) ---
    if show_nlg:
        nlg_layer = folium.FeatureGroup(name="🔴 Night-Light Gap Alerts", show=True)
        zone_preds  = plan["zone_preds"]
        grid_lat    = plan["grid_lat"]
        grid_lon    = plan["grid_lon"]
        viirs       = plan["viirs_dnb"]
        road_dist   = plan["road_dist_m"]
        delta_ndbi  = plan["delta_ndbi"]
        vv_db       = plan["vv_db"]
        sprawl_prob = plan["sprawl_prob"]
        half_cell   = 0.00113

        for r in range(n):
            for c in range(n):
                if zone_preds[r, c] == 2:
                    clat  = float(grid_lat[r, c])
                    clon  = float(grid_lon[r, c])
                    risk  = float(sprawl_prob[r, c])
                    bounds = [
                        [clat - half_cell, clon - half_cell],
                        [clat + half_cell, clon + half_cell],
                    ]
                    deficit = max(0, 10 - float(viirs[r, c]))
                    popup_html = f"""<div style="background:#0F172A;color:#E2E8F0;padding:12px;
                                               border-radius:8px;font-family:monospace;font-size:11px;
                                               border:1px solid #EF4444;min-width:230px">
                      <b style="color:#EF4444;font-size:13px">⚠ NIGHT-LIGHT GAP ALERT</b><br>
                      <hr style="border-color:#1E293B;margin:6px 0">
                      📍 {clat:.5f}, {clon:.5f}<br>
                      🏗 Built-Up Growth (ΔNDBI): <b style="color:#F59E0B">{delta_ndbi[r,c]:.3f}</b><br>
                      📡 Radar VV: <b>{vv_db[r,c]:.1f} dB</b><br>
                      💡 VIIRS Night Light: <b style="color:#EF4444">{viirs[r,c]:.1f} nW/cm²/sr</b><br>
                      🛣 Road Distance: <b>{road_dist[r,c]:.0f} m</b><br>
                      ⚡ Electricity Deficit: <b style="color:#EF4444">{deficit:.1f}</b><br>
                      📈 2027-28 Risk: <b style="color:#F59E0B">{risk:.1%}</b>
                    </div>"""
                    folium.Rectangle(
                        bounds=bounds,
                        color="#EF4444",
                        fill=True,
                        fill_color="#EF4444",
                        fill_opacity=0.65,
                        weight=1,
                        popup=folium.Popup(popup_html, max_width=300),
                    ).add_to(nlg_layer)
        nlg_layer.add_to(m)

    # --- 2027–28 Sprawl Forecast Heatmap ---
    if show_forecast:
        forecast_layer = folium.FeatureGroup(name="🟠 2027–28 Migration Forecast", show=True)
        grid_lat    = plan["grid_lat"]
        grid_lon    = plan["grid_lon"]
        sprawl_prob = plan["sprawl_prob"]

        heat_data = [
            [float(grid_lat[r, c]), float(grid_lon[r, c]), float(sprawl_prob[r, c])]
            for r in range(n) for c in range(n)
            if sprawl_prob[r, c] > 0.35
        ]
        if heat_data:
            HeatMap(
                heat_data,
                radius=20, blur=15, max_zoom=14,
                gradient={"0.4": "#78350F", "0.65": "#F59E0B", "1.0": "#FDE68A"},
                name="Forecast Heatmap",
            ).add_to(forecast_layer)
        forecast_layer.add_to(m)

    # --- AI Auto-Routed Corridors ---
    if show_corridors and plan.get("corridors"):
        corridor_layer = folium.FeatureGroup(name="🔵 AI Bus & Power Corridors", show=True)
        for corr in plan["corridors"]:
            geom = corr["geojson"]["geometry"]
            coords_latlng = [[pt[1], pt[0]] for pt in geom["coordinates"]]
            popup_html = f"""<div style="background:#0F172A;color:#E2E8F0;padding:12px;
                                       border-radius:8px;font-family:monospace;font-size:11px;
                                       border:1px solid #06B6D4;min-width:240px">
              <b style="color:#06B6D4;font-size:13px">🛣 AI CORRIDOR — RANK #{corr['rank']}</b><br>
              <hr style="border-color:#1E293B;margin:6px 0">
              📏 Length: <b style="color:#38BDF8">{corr['corridor_km']:.1f} km</b><br>
              🏠 Households Served: <b style="color:#22C55E">{corr['served_hh']:,}</b><br>
              💰 Est. Cost (Early): <b style="color:#06B6D4">₹{corr['cost_cr']:.1f} Cr</b><br>
              📈 Sprawl Risk: <b style="color:#F59E0B">{corr['risk_score']:.1%}</b><br>
              📍 From: {corr['source_lat']:.4f}, {corr['source_lon']:.4f}<br>
              📍 To: {corr['target_lat']:.4f}, {corr['target_lon']:.4f}
            </div>"""
            folium.PolyLine(
                locations=coords_latlng,
                color="#06B6D4",
                weight=4,
                opacity=0.9,
                dash_array=None,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Corridor #{corr['rank']} — {corr['corridor_km']:.1f} km | ₹{corr['cost_cr']:.1f} Cr",
            ).add_to(corridor_layer)
            # Glowing effect — duplicate with higher opacity and thinner
            folium.PolyLine(
                locations=coords_latlng,
                color="#38BDF8",
                weight=1.5,
                opacity=0.5,
            ).add_to(corridor_layer)
            # Source marker
            folium.CircleMarker(
                location=[corr["source_lat"], corr["source_lon"]],
                radius=7, color="#EF4444", fill=True, fill_color="#EF4444", fill_opacity=0.9,
                tooltip=f"NLG Source #{corr['rank']}",
            ).add_to(corridor_layer)
            # Target marker (road node)
            folium.CircleMarker(
                location=[corr["target_lat"], corr["target_lon"]],
                radius=5, color="#06B6D4", fill=True, fill_color="#06B6D4", fill_opacity=0.9,
                tooltip="Existing Road Node",
            ).add_to(corridor_layer)
        corridor_layer.add_to(m)

    # Centre marker
    folium.Marker(
        location=[lat, lon],
        popup=f"Analysis Centre ({lat:.4f}, {lon:.4f})",
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
    ).add_to(m)

    MiniMap(toggle_display=True, tile_layer="CartoDB dark_matter").add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


# ============================================================================
# PLOTLY CHARTS
# ============================================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0A1020",
    plot_bgcolor="#0F172A",
    font=dict(color="#E2E8F0", family="Inter, Segoe UI, sans-serif", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
    yaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
)


def _chart_nlg_proof(plan: dict) -> go.Figure:
    """Chart: Daytime concrete growth vs. night-light deficit."""
    nc = plan["n_cells"]
    zone_flat  = plan["zone_preds"].ravel()
    ndbi_flat  = plan["delta_ndbi"].ravel()
    viirs_flat = plan["viirs_dnb"].ravel()
    zone_names = [ZONE_SHORT.get(int(z), "?") for z in zone_flat]

    df = pd.DataFrame({
        "Delta NDBI (Concrete Growth)": ndbi_flat,
        "VIIRS Night Light (nW/cm²/sr)": viirs_flat,
        "Zone": zone_names,
        "Zone_id": zone_flat.astype(int),
    })

    color_map = {ZONE_SHORT[k]: v for k, v in ZONE_COLORS.items()}

    fig = px.scatter(
        df,
        x="Delta NDBI (Concrete Growth)",
        y="VIIRS Night Light (nW/cm²/sr)",
        color="Zone",
        color_discrete_map=color_map,
        opacity=0.55,
        title="Night-Light Gap Evidence: Daytime Build-Up vs. Night Electricity",
        labels={"Delta NDBI (Concrete Growth)": "ΔNDBI (Built-Up Growth 2021–2026)",
                "VIIRS Night Light (nW/cm²/sr)": "VIIRS Night-Light Intensity"},
    )
    fig.add_hline(y=5, line_dash="dash", line_color="#EF4444",
                  annotation_text="Night-Light Gap Threshold (<5 nW)", annotation_font_color="#EF4444")
    fig.add_vline(x=0.05, line_dash="dash", line_color="#F59E0B",
                  annotation_text="Built-Up Threshold (ΔNDBI>0.05)", annotation_font_color="#F59E0B")
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=True, legend=dict(
        bgcolor="#0F172A", bordercolor="#1E293B"))
    return fig


def _chart_radar_split(plan: dict) -> go.Figure:
    """Chart: Radar VV backscatter distribution by zone — building height proxy."""
    zone_flat = plan["zone_preds"].ravel()
    vv_flat   = plan["vv_db"].ravel()

    fig = go.Figure()
    for z_id, z_name in ZONE_LABELS.items():
        mask = (zone_flat == z_id)
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Violin(
            y=vv_flat[mask],
            name=ZONE_SHORT[z_id],
            fillcolor=ZONE_COLORS[z_id],
            line_color=ZONE_COLORS[z_id],
            opacity=0.75,
            box_visible=True,
            meanline_visible=True,
            points="outliers",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Sentinel-1 Radar VV Backscatter by Zone (dB) — Building Height Proxy",
        yaxis_title="VV Backscatter (dB)",
        showlegend=True,
        legend=dict(bgcolor="#0F172A", bordercolor="#1E293B"),
    )
    return fig


def _chart_sprawl_forecast(plan: dict) -> go.Figure:
    """Chart: 2027–28 sprawl probability spatial distribution."""
    nc  = plan["n_cells"]
    sp  = plan["sprawl_prob"]
    lat = plan["grid_lat"]
    lon = plan["grid_lon"]

    fig = go.Figure(go.Contour(
        z=sp,
        x=lon[0, :],
        y=lat[:, 0],
        colorscale=[[0, "#0F172A"], [0.4, "#78350F"], [0.7, "#F59E0B"], [1, "#FDE68A"]],
        contours=dict(start=0.2, end=1.0, size=0.1),
        colorbar=dict(title="Risk", tickfont=dict(color="#E2E8F0"),
                      titlefont=dict(color="#E2E8F0"),
                      bgcolor="#0F172A", bordercolor="#1E293B"),
        line_smoothing=0.85,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="2027–28 AI Migration Gravity Forecast — Sprawl Risk Heatmap",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )
    return fig


def _chart_feature_importance(plan: dict) -> go.Figure:
    """Chart: AI model feature importances."""
    fi = plan.get("feat_importances", {})
    if not fi:
        return go.Figure()
    labels = list(fi.keys())
    values = [fi[k] for k in labels]
    # Sort descending
    pairs = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*pairs) if pairs else ([], [])

    friendly = {
        "delta_ndbi":  "ΔNDBI Built-Up Growth",
        "ndvi_2026":   "NDVI Vegetation 2026",
        "ndbi_2026":   "NDBI Built-Up 2026",
        "vv_db":       "SAR VV Backscatter",
        "vh_db":       "SAR VH Backscatter",
        "viirs_dnb":   "VIIRS Night-Lights",
        "road_dist_m": "Road Distance",
        "water_mask":  "Water / Green Mask",
    }
    labels = [friendly.get(l, l) for l in labels]
    colors = ["#06B6D4" if i == 0 else "#1E3A5F" for i in range(len(labels))]

    fig = go.Figure(go.Bar(
        x=list(values), y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#E2E8F0"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"AI Model Feature Importances (CV Accuracy: {plan.get('cv_accuracy', 0):.1%})",
        xaxis_title="Importance Score",
        yaxis=dict(autorange="reversed", gridcolor="#1E293B"),
    )
    return fig


def _chart_roi_comparison(roi: dict) -> go.Figure:
    """Chart: Act Now vs. Wait 3 Years cost comparison bar."""
    if not roi or roi["total_funded_cost_cr"] == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Act Now (Early Corridors)", "Wait 3 Years (Retrofit)"],
        y=[roi["total_funded_cost_cr"], roi["deferred_cost_cr"]],
        marker_color=["#06B6D4", "#EF4444"],
        text=[f"₹{roi['total_funded_cost_cr']:.1f} Cr",
              f"₹{roi['deferred_cost_cr']:.1f} Cr"],
        textposition="outside",
        textfont=dict(color="#E2E8F0", size=14),
        width=0.4,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Act Now vs. Wait 3 Years — Net Savings: ₹{roi['net_savings_cr']:.1f} Cr (${roi['net_savings_usd_m']:.1f} M)",
        yaxis_title="Cost (₹ Crore)",
        showlegend=False,
    )
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================

def _render_sidebar() -> tuple[float, float, str, float, dict]:
    """Render sidebar controls; return (lat, lon, hub_name, budget_cr, toggles)."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:8px 0 12px 0">
          <div style="font-size:1.5rem">🛰️</div>
          <div style="font-size:0.9rem;font-weight:800;color:#06B6D4;letter-spacing:1px">
            URBANPULSE AI
          </div>
          <div style="font-size:0.65rem;color:#475569;letter-spacing:2px;text-transform:uppercase">
            Control Center
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-head">📍 Analysis Location</div>', unsafe_allow_html=True)

        hub_names = ["— Live Custom Search —"] + list(PRESET_HUBS.keys())
        selected_hub = st.selectbox(
            "Karnataka Preset Hub",
            hub_names,
            index=1,
            help="Select a pre-cached preset for instant loading, or use Live Custom Search below.",
        )

        lat, lon, hub_name = None, None, selected_hub

        if selected_hub != "— Live Custom Search —":
            hub = PRESET_HUBS[selected_hub]
            lat, lon = hub["lat"], hub["lon"]
            st.markdown(f"""
            <div style="background:#0A1020;border:1px solid #1E3A5F;border-radius:8px;
                        padding:8px 12px;margin-top:6px;font-size:0.76rem;color:#64748B">
              <span style="color:#06B6D4">●</span> {hub['description']}<br>
              <span style="color:#38BDF8">Authority:</span> {hub['authority']}<br>
              <span style="color:#38BDF8">Coordinates:</span> {lat:.4f}°N, {lon:.4f}°E
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-head">🔍 Live Satellite Scan</div>', unsafe_allow_html=True)
            query = st.text_input(
                "City / Area or lat,lon",
                placeholder="e.g. Hubli, Karnataka  or  15.3647, 75.1240",
                key="location_query",
            )
            if st.button("🛰️  Run Live Satellite & AI Scan", key="scan_btn"):
                if query.strip():
                    with st.spinner("Geocoding location…"):
                        coords = _geocode_location(query.strip())
                    if coords:
                        lat, lon = coords
                        st.session_state["custom_lat"] = lat
                        st.session_state["custom_lon"] = lon
                        st.success(f"📍 Located: {lat:.4f}°N, {lon:.4f}°E")
                    else:
                        st.error("Could not locate that place. Try lat,lon format.")
                else:
                    st.warning("Please enter a location.")

            if "custom_lat" in st.session_state and selected_hub == "— Live Custom Search —":
                lat = st.session_state["custom_lat"]
                lon = st.session_state["custom_lon"]
                hub_name = f"Custom ({lat:.3f}°N, {lon:.3f}°E)"

        st.markdown('<div class="section-head">💰 Policy Simulator</div>', unsafe_allow_html=True)
        budget_cr = st.slider(
            "Early Intervention Budget (₹ Crore)",
            min_value=10,
            max_value=500,
            value=50,
            step=5,
            format="₹ %d Cr",
            key="budget_slider",
        )

        st.markdown('<div class="section-head">🗺️ Map Layers</div>', unsafe_allow_html=True)
        toggles = {
            "zones":     st.toggle("4-Zone AI Classification",     value=True,  key="tog_zones"),
            "nlg":       st.toggle("Night-Light Gap Alerts",        value=True,  key="tog_nlg"),
            "forecast":  st.toggle("2027–28 Migration Forecast",    value=True,  key="tog_forecast"),
            "corridors": st.toggle("AI Bus & Power Corridors",      value=True,  key="tog_corridors"),
        }

        st.markdown("""
        <hr style="border-color:#1E293B;margin:14px 0 10px 0">
        <div style="font-size:0.68rem;color:#334155;text-align:center;line-height:1.6">
          Aeromania '26 Mach-A-Thon<br>
          Track A: Migration & Urbanization<br>
          <span style="color:#06B6D4">Boeing Aerospace Division</span>
        </div>
        """, unsafe_allow_html=True)

    return lat, lon, hub_name, budget_cr, toggles


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    lat, lon, hub_name, budget_cr, toggles = _render_sidebar()

    # --- Header ---
    st.markdown("""
    <div class="header-strip">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap">
        <div>
          <div class="header-title">🛰️ UrbanPulse AI</div>
          <div class="header-subtitle">Orbit-to-Economy Migration & Sprawl Intelligence · Track A</div>
          <div style="margin-top:8px">
            <span class="sensor-pill active">▲ SENTINEL-2 OPTICAL</span>
            <span class="sensor-pill active">◈ SENTINEL-1 RADAR</span>
            <span class="sensor-pill active">★ VIIRS NIGHT-LIGHTS</span>
            <span class="sensor-pill active">⬡ OSM GRAPH</span>
          </div>
        </div>
        <div style="text-align:right;font-size:0.72rem;color:#475569;margin-top:4px">
          Aeromania '26 · Boeing Judges<br>
          <span style="color:#22C55E">◉ LIVE</span>  Planetary Computer STAC
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Value Banner ---
    st.markdown("""
    <div class="value-banner">
      <strong>💼 Who Pays & Economic Value:</strong>
      Urban Development Authorities (BDA · TUDA · MUDA · NMMC) & Municipal Corporations pay for
      UrbanPulse AI to <strong>lock in transit & power corridors before encroachment</strong> —
      saving ₹ Crores by acting early instead of demolishing and retrofitting 3 years later at
      <strong>3.4× the cost.</strong>
    </div>
    """, unsafe_allow_html=True)

    # --- Guard: no location selected yet ---
    if lat is None or lon is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#334155">
          <div style="font-size:3rem">🛰️</div>
          <div style="font-size:1.1rem;color:#38BDF8;margin:10px 0">Select a Karnataka Hub or search a custom location</div>
          <div style="font-size:0.85rem">Use the sidebar to pick a preset or type any city / lat,lon coordinates.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- Run pipeline (with spinner) ---
    with st.spinner(f"🛰️  Running satellite fusion + AI pipeline for {hub_name} …"):
        try:
            plan = _run_full_pipeline(lat, lon, budget_cr)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            return

    roi = plan.get("roi", {})

    # --- KPI Hero Cards ---
    st.markdown('<div class="section-head">📊 Mission-Critical KPIs</div>', unsafe_allow_html=True)

    new_built   = plan.get("new_built_sqkm", 0)
    n_nlg       = plan.get("n_nlg_alerts", 0)
    nlg_area    = plan.get("nlg_area_sqkm", 0)
    influx      = plan.get("influx_2027_28", 0)
    influx_cov  = roi.get("influx_covered", 0)
    unmit_inf   = roi.get("unmitigated_influx", influx)
    sprawl_prev = roi.get("sprawl_prevented_sqkm", 0)
    cov_pct     = roi.get("coverage_ratio", 0)
    net_sav_cr  = roi.get("net_savings_cr", 0)
    net_sav_m   = roi.get("net_savings_usd_m", 0)

    c1, c2, c3, c4 = st.columns(4)
    kpi_cards = [
        (c1, "cyan",  f"{new_built:.1f} km²",                      "New Built-Up Growth",      f"Prevented Slum Sprawl: {sprawl_prev:.1f} km²"),
        (c2, "red",   f"{n_nlg:,}",                                 "Night-Light Gap Alerts",   f"{nlg_area:.1f} km² underserved · {roi.get('n_funded', 0)} corridors"),
        (c3, "amber", f"{unmit_inf/1000:.1f}K",                     "2027–28 Predicted Influx", f"Total {influx/1000:.1f}K · {influx_cov/1000:.1f}K sheltered"),
        (c4, "green", f"₹{net_sav_cr:.0f} Cr / ${net_sav_m:.1f}M", "Net Planner Savings",      f"Act now vs. 3.4× retrofit · {cov_pct:.0f}% cov"),
    ]
    for col, color, val, label, delta in kpi_cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card {color}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {color}">{val}</div>
              <div class="kpi-delta" style="color:#64748B">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Centerpiece Map ---
    st.markdown(f'<div class="section-head">🗺️ Geospatial Command Map — {hub_name}</div>',
                unsafe_allow_html=True)

    with st.spinner("Rendering interactive map …"):
        fmap = _build_map(
            plan,
            show_zones=toggles["zones"],
            show_nlg=toggles["nlg"],
            show_forecast=toggles["forecast"],
            show_corridors=toggles["corridors"],
        )
    st_folium(fmap, width=None, height=580, returned_objects=[])

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3 = st.tabs([
        "📡 Multi-Sensor Proof & AI Accuracy",
        "🗂️ Planner Blueprint & Export",
        "🏗️ Orbit-to-Economy Architecture",
    ])

    # --- Tab 1: Multi-Sensor Proof ---
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(_chart_nlg_proof(plan), use_container_width=True)
        with col_b:
            st.plotly_chart(_chart_radar_split(plan), use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(_chart_sprawl_forecast(plan), use_container_width=True)
        with col_d:
            st.plotly_chart(_chart_feature_importance(plan), use_container_width=True)

        # Accuracy metric
        cv_acc = plan.get("cv_accuracy", 0)
        st.markdown(f"""
        <div style="background:#0F172A;border:1px solid #1E3A5F;border-radius:8px;
                    padding:14px 18px;margin-top:8px;display:flex;gap:40px;flex-wrap:wrap">
          <div>
            <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px">
              Classifier CV Accuracy
            </div>
            <div style="font-size:1.8rem;font-weight:800;color:#22C55E">{cv_acc:.1%}</div>
            <div style="font-size:0.72rem;color:#475569">5-Fold StratifiedKFold | GBM</div>
          </div>
          <div>
            <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px">
              Grid Resolution
            </div>
            <div style="font-size:1.8rem;font-weight:800;color:#06B6D4">250 m</div>
            <div style="font-size:0.72rem;color:#475569">48 × 48 = 2,304 cells / 12 km²</div>
          </div>
          <div>
            <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px">
              Sprawl Risk (Mean / Max)
            </div>
            <div style="font-size:1.8rem;font-weight:800;color:#F59E0B">
              {plan.get('sprawl_prob_mean', 0):.2f} / {plan.get('sprawl_prob_max', 0):.2f}
            </div>
            <div style="font-size:0.72rem;color:#475569">0.0 → 1.0 (Spatial RF Regressor)</div>
          </div>
          <div>
            <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px">
              Data Sources
            </div>
            <div style="font-size:1.3rem;font-weight:800;color:#38BDF8">S2 · S1 · VIIRS · OSM</div>
            <div style="font-size:0.72rem;color:#475569">Microsoft Planetary Computer STAC</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Tab 2: Planner Blueprint ---
    with tab2:
        corridors = plan.get("corridors", [])

        # ROI comparison chart
        st.plotly_chart(_chart_roi_comparison(roi), use_container_width=True)

        # ROI summary cards
        r1, r2, r3, r4 = st.columns(4)
        for col, label, val, color in [
            (r1, "Early Cost",    f"₹{roi.get('total_funded_cost_cr', 0):.1f} Cr", "#06B6D4"),
            (r2, "Deferred Cost", f"₹{roi.get('deferred_cost_cr', 0):.1f} Cr",    "#EF4444"),
            (r3, "HH Connected",  f"{roi.get('hh_connected', 0):,}",              "#22C55E"),
            (r4, "Sprawl Saved",  f"{roi.get('sprawl_prevented_sqkm', 0):.1f} km²", "#F59E0B"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#0F172A;border:1px solid #1E293B;border-radius:8px;
                            padding:12px;text-align:center">
                  <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase">{label}</div>
                  <div style="font-size:1.5rem;font-weight:800;color:{color}">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        # Priority intervention table
        st.markdown('<div class="section-head">🏆 Top-5 Priority Intervention Corridors</div>',
                    unsafe_allow_html=True)

        if corridors:
            df_tbl = pd.DataFrame([{
                "Rank":          f"#{c['rank']}",
                "Source Lat":    f"{c['source_lat']:.5f}",
                "Source Lon":    f"{c['source_lon']:.5f}",
                "Target Lat":    f"{c['target_lat']:.5f}",
                "Target Lon":    f"{c['target_lon']:.5f}",
                "Length (km)":   f"{c['corridor_km']:.1f}",
                "HH Served":     f"{c['served_hh']:,}",
                "Cost Early (₹Cr)": f"{c['cost_cr']:.2f}",
                "Cost Deferred (₹Cr)": f"{c['cost_cr'] * 3.4:.2f}",
                "Savings (₹Cr)": f"{c['cost_cr'] * 2.4:.2f}",
                "Risk Score":    f"{c['risk_score']:.3f}",
            } for c in corridors])

            st.dataframe(
                df_tbl,
                use_container_width=True,
                hide_index=True,
            )

            # CSV download
            csv_bytes = df_tbl.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Planner Blueprint (CSV)",
                data=csv_bytes,
                file_name=f"urbanpulse_blueprint_{hub_name.replace(' ', '_')[:30]}.csv",
                mime="text/csv",
                key="dl_csv",
                help="Download the ranked intervention table as a CSV for the planning authority.",
            )
        else:
            st.info("No corridors computed yet.")

    # --- Tab 3: Architecture ---
    with tab3:
        st.markdown("""
        <div style="margin-bottom:12px;font-size:0.82rem;color:#64748B;line-height:1.6">
          How <b style="color:#06B6D4">UrbanPulse AI</b> feeds directly into the unified
          <b>5-person Track A platform</b> — from raw satellite orbit to economic planning outputs.
        </div>
        """, unsafe_allow_html=True)

        arch_items = [
            ("🛰️ Planetary Computer STAC",
             "Microsoft's public, zero-API-key STAC catalog. Live windowed reads of Sentinel-2 L2A, Sentinel-1 GRD, and VIIRS DNB tiles without downloading multi-GB files.",
             "#164E63"),
            ("📡 data_pipeline.py — Multi-Sensor Fusion",
             "Fetches & aligns 4 layers onto a 250 m × 250 m, 12 km × 12 km grid: ΔNDBI (concrete growth), SAR VV/VH backscatter (building height), VIIRS night-light gap, OSM road-distance + water mask.",
             "#1E3A5F"),
            ("🤖 ml_engine.py — AI Classifier + Forecaster",
             "Gradient Boosting 4-Zone Classifier (Tech Core · Legal Suburb · Night-Light Gap · Green Buffer) + Spatial RF Regressor for 2027–28 migration gravity sprawl risk map. Physics-informed training labels derived from satellite statistics.",
             "#1E3A5F"),
            ("🗺️ planner_engine.py — Corridor Router + ROI Sim",
             "NetworkX Dijkstra auto-routes cheapest bus-road + power-line corridors from NLG settlements to existing OSM arterial roads, respecting water/green cost penalties. Act Now vs. 3-Year Deferral ROI simulation in ₹ Crore and $ Million.",
             "#1E3A5F"),
            ("🖥️ app.py — Aerospace Command Center UI",
             "Streamlit + Folium (dark CartoDB basemap) + Plotly charts. Glassmorphic aerospace navy theme, live KPI cards, full-width interactive map with styled HTML popups, 3 executive tabs, and 1-click CSV export.",
             "#1E3A5F"),
            ("🏛️ Urban Development Authority Integration",
             "Outputs are BDA / TUDA / MUDA / NMMC-ready: GeoJSON corridor paths + rupee ROI reports that planning authorities can use directly in corridor reservation orders (Section 11A, KTCP Act).",
             "#14532D"),
        ]

        for i, (name, desc, border_color) in enumerate(arch_items):
            st.markdown(f"""
            <div class="arch-card" style="border-color:{border_color}">
              <div class="module-name">{name}</div>
              <div class="module-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if i < len(arch_items) - 1:
                st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:16px;background:#0F172A;border:1px solid #1E3A5F;border-radius:8px;
                    padding:14px 18px;font-size:0.78rem;color:#64748B;line-height:1.7">
          <b style="color:#06B6D4">Team Track A — 5 Modules:</b><br>
          🛰 UrbanPulse AI (Migration & Urbanization) ·
          🌊 FloodSense (Disaster Risk) ·
          🌾 AgroSat (Agricultural Monitoring) ·
          🏗 InfraScope (Infrastructure Change) ·
          📡 TrafficOrbit (Road & Rail Demand)
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

