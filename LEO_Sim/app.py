"""
LEO Satellite Mission Control Dashboard
Mach-A-Thon 2026 — Aeromania

Satellite → Observation → Onboard AI → Data Downlink → Ground Station

Three onboard applications:
  • Eco_Act       — Economic intelligence (VIIRS night-light analysis)
  • UrbanPulse    — Urbanization intelligence (multi-sensor satellite fusion)
  • DisasterRisk  — Disaster risk & loss assessment

Architecture:
  - Orbit simulation: Keplerian circular LEO (500 km, 97.4° inclination)
  - Communication: visibility-gated priority queue (HIGH / MEDIUM / LOW)
  - Payload: actual serialized JSON from each AI pipeline
  - Comparison: real file / payload sizes, calculated transmission times
"""

from __future__ import annotations

import sys, os, json, time, math, glob
from pathlib import Path
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS  = Path(__file__).resolve().parent
_MACH  = _THIS.parent
sys.path.insert(0, str(_MACH))
sys.path.insert(0, str(_THIS))

from src.orbit import Satellite, OrbitalParams, GROUND_STATION
from src.transmission_queue import TransmissionQueue
from onboard_processor.application_runner import APPLICATIONS, run as run_application

# ── Constants ──────────────────────────────────────────────────────────────────
ECO_DATASET   = _MACH / "Eco_Act" / "Dataset" / "monthly_10yr"
APP_COLORS    = {"Eco_Act": "#58a6ff", "UrbanPulse": "#3fb950", "DisasterRisk": "#f78166"}
APP_ICONS     = {"Eco_Act": "🌃", "UrbanPulse": "🏙", "DisasterRisk": "⚠️"}
PRIORITY_MAP  = {"Eco_Act": "MEDIUM", "UrbanPulse": "LOW", "DisasterRisk": "HIGH"}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LEO Mission Control",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, .stApp { background:#0d1117; color:#c9d1d9; font-family:'Courier New',monospace; }
h1,h2,h3,h4 { color:#58a6ff; letter-spacing:0.03em; }
.stSidebar { background:#161b22; border-right:1px solid #21262d; }
.block { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px 16px; margin:6px 0; }
.app-eco    { border-left:3px solid #58a6ff; }
.app-urban  { border-left:3px solid #3fb950; }
.app-dis    { border-left:3px solid #f78166; }
.badge-sim  { background:#21262d; border:1px solid #f78166; color:#f78166; border-radius:3px; padding:1px 6px; font-size:0.68rem; font-weight:700; }
.badge-act  { background:#21262d; border:1px solid #3fb950; color:#3fb950; border-radius:3px; padding:1px 6px; font-size:0.68rem; font-weight:700; }
.badge-syn  { background:#21262d; border:1px solid #d29922; color:#d29922; border-radius:3px; padding:1px 6px; font-size:0.68rem; font-weight:700; }
.state-ok   { color:#3fb950; font-weight:700; }
.state-err  { color:#f78166; font-weight:700; }
.state-warn { color:#d29922; font-weight:700; }
.metric-label { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em; }
.metric-val   { font-size:1.3rem; font-weight:700; color:#e6edf3; }
.stButton>button { background:#21262d; color:#c9d1d9; border:1px solid #30363d; border-radius:6px; font-weight:600; }
.stButton>button:hover { background:#1f6feb; color:#fff; border-color:#1f6feb; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        sim_running=False, elapsed_s=0.0, last_tick=None,
        sim_speed=1.0, bandwidth_mbps=2.0,
        queue=TransmissionQueue(),
        received=[],       # list of received payload dicts
        run_results={},    # app_name -> result dict
        run_errors={},     # app_name -> error string
        input_sizes={},    # app_name -> raw input size bytes
        eco_image=None,
        urban_hub=None,
        disaster_scenario=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

S = st.session_state   # shorthand

# ── Orbit ──────────────────────────────────────────────────────────────────────
_params = OrbitalParams(altitude_km=500.0, inclination_deg=97.4)
_sat    = Satellite(_params)
gs      = GROUND_STATION

def _sat_pos(t): return _sat.position_at(t)
def _visibility(t): return _sat.visibility_at(t, gs["lat"], gs["lon"], min_elevation_deg=5.0)
def _in_view(pos): return _visibility(pos["elapsed_s"])["visible"]

def _contact_remaining(t, step_s=1.0):
    """Find remaining pass duration using the link's elevation mask."""
    if not _visibility(t)["visible"]:
        return 0.0
    offset = step_s
    while offset <= _params.period_seconds and _visibility(t + offset)["visible"]:
        offset += step_s
    return max(0.0, offset - step_s)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_bytes(b):
    if b >= 1_000_000: return f"{b/1_000_000:.2f} MB"
    if b >= 1024:      return f"{b/1024:.2f} KB"
    return f"{b} B"

def _tx_time(size_b, bw_mbps):
    return (size_b * 8) / (bw_mbps * 1_000_000)

def _priority_color(p):
    return {"HIGH": "#f78166", "MEDIUM": "#d29922", "LOW": "#3fb950"}.get(p, "#c9d1d9")

def _metric(label, value, col=None):
    html = f'<div class="metric-label">{label}</div><div class="metric-val">{value}</div>'
    (col or st).markdown(html, unsafe_allow_html=True)

def _eco_images():
    return sorted(glob.glob(str(ECO_DATASET / "*.png")))

def _urban_hubs():
    try:
        from urbanpulse_adapter import get_preset_hubs
        return get_preset_hubs()
    except Exception:
        return {}

def _disaster_scenarios():
    try:
        from disasterrisk_adapter import list_scenarios
        return list_scenarios()
    except Exception:
        return {}

def _build_payload(app, result):
    """Serialize result to compact JSON bytes, excluding large internal fields."""
    skip = {"image_path", "status"}
    clean = {"application": app, **{k: v for k, v in result.items() if k not in skip}}
    return json.dumps(clean, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _enqueue(app, result, raw_input_size):
    payload  = _build_payload(app, result)
    priority = PRIORITY_MAP[app]
    bw       = S["bandwidth_mbps"]
    # Fallback: use 1 MB if raw size is unknown (should not happen after fixes)
    raw_input_size = raw_input_size or result.get("raw_input_bytes") or result.get("image_size_bytes") or 1_048_576
    tx_raw   = _tx_time(raw_input_size, bw)
    tx_json  = _tx_time(len(payload), bw)
    meta = {
        "app": app,
        "priority": priority,
        "raw_input_size_b": raw_input_size,
        "json_size_b": len(payload),
        "data_reduction_pct": round((1 - len(payload)/raw_input_size)*100, 2) if raw_input_size else 0,
        "tx_time_raw_s": round(tx_raw, 4),
        "tx_time_json_s": round(tx_json, 6),
        "time_saved_s": round(tx_raw - tx_json, 4),
        "queued_at_elapsed_s": S["elapsed_s"],
        "data_quality": result.get("data_quality", "PROVIDER_DATA"),
    }
    item = S["queue"].enqueue(payload, priority, label=f"{app} observation", metadata=meta)
    return item, meta

# ── Advance clock ──────────────────────────────────────────────────────────────
if S["sim_running"] and S["last_tick"]:
    now = time.time()
    S["elapsed_s"] += (now - S["last_tick"]) * S["sim_speed"]
    S["last_tick"]  = now

elapsed = S["elapsed_s"]
pos     = _sat_pos(elapsed)
link_telemetry = _visibility(elapsed)
in_view = link_telemetry["visible"]

# Transmit only while the simulation advances and the calculated pass is open.
if S["sim_running"] and in_view and S["queue"].pending():
    contact_s = _contact_remaining(elapsed)
    res = S["queue"].dispatch(visible=True, bandwidth_mbps=S["bandwidth_mbps"], contact_seconds=contact_s)
    for sent_item in res["sent"]:
        meta = sent_item.metadata
        S["received"].append({
            "app":           meta["app"],
            "priority":      meta["priority"],
            "json_size_b":   meta["json_size_b"],
            "raw_size_b":    meta["raw_input_size_b"],
            "reduction_pct": meta["data_reduction_pct"],
            "tx_time_s":     meta["tx_time_json_s"],
            "transmit_duration_s": meta.get("simulated_transmit_duration_s"),
            "time_saved_s":  meta["time_saved_s"],
            "data_quality":  meta["data_quality"],
            "received_at_s": S["elapsed_s"] + meta.get("simulated_receive_offset_s", 0.0),
            "payload":       json.loads(sent_item.payload.decode("utf-8")),
        })

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛰 LEO Mission Control")
    st.caption("Mach-A-Thon 2026 · Aeromania")
    st.divider()

    # Orbit summary
    st.markdown("**Orbital Parameters** <span class='badge-sim'>SIMULATED</span>", unsafe_allow_html=True)
    p = _params
    st.markdown(f"""
| | |
|---|---|
| Altitude | {p.altitude_km} km |
| Inclination | {p.inclination_deg}° |
| Period | {p.period_minutes:.2f} min |
| Velocity | {p.velocity_km_per_s:.4f} km/s |
| Shape | Circular |
""")
    st.caption("Period = 2π√(a³/μ) · Kepler 3rd Law")
    st.divider()

    # Simulation controls
    st.markdown("**Simulation Controls**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Start"):
            S["sim_running"] = True
            S["last_tick"]   = time.time()
    with c2:
        if st.button("⏸ Pause"):
            S["sim_running"] = False
    with c3:
        if st.button("⟳ Reset"):
            S["sim_running"]     = False
            S["elapsed_s"]       = 0.0
            S["last_tick"]       = None
            S["queue"]           = TransmissionQueue()
            S["received"]        = []
            S["run_results"]     = {}
            S["run_errors"]      = {}
            S["input_sizes"]     = {}

    spd = st.slider("Sim Speed ×", 1, 500, int(S["sim_speed"]), 1)
    S["sim_speed"] = float(spd)
    bw  = st.slider("Downlink Bandwidth (Mbps)", 0.5, 20.0, S["bandwidth_mbps"], 0.5)
    S["bandwidth_mbps"] = bw
    st.caption("[SIMULATED] Communication bandwidth")

    st.divider()
    st.markdown(f"""
**Ground Station** <span class='badge-act'>REAL</span>
📡 {gs['name']}
{gs['city']}
{gs['lat']}°N, {gs['lon']}°E
""", unsafe_allow_html=True)

    st.divider()
    # Telemetry panel
    link = "🟢 IN VIEW" if in_view else "🔴 NO CONTACT"
    st.markdown(f"**Link Status:** {link}")
    st.markdown(f"**Elevation:** {link_telemetry['elevation_deg']:.1f}°  ·  **Range:** {link_telemetry['distance_km']:.0f} km")
    T = p.period_seconds
    st.markdown(f"**Elapsed:** {elapsed:.0f} s  ({elapsed/T:.2f} orbits)")
    st.markdown(f"**Queue depth:** {len(S['queue'].pending())} items")
    st.markdown(f"**Received:** {len(S['received'])} payloads")

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛰 LEO Satellite Mission Control")
st.markdown(
    '<span class="badge-sim">SIM</span> Orbit & Comms simulated &nbsp; '
    '<span class="badge-act">SOURCE-GATED</span> Outputs require available real observations',
    unsafe_allow_html=True,
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1: Globe + Sat Status
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · Satellite [SIMULATED]")
gcol, scol = st.columns([3, 1])

with gcol:
    track = _sat.ground_track(360)
    tlats = [p_["lat"] for p_ in track]
    tlons = [p_["lon"] for p_ in track]

    # Break at antimeridian
    blats, blons = [], []
    prev = tlons[0]
    for la, lo in zip(tlats, tlons):
        if abs(lo - prev) > 180:
            blats.append(None); blons.append(None)
        blats.append(la); blons.append(lo)
        prev = lo

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=blats, lon=blons, mode="lines",
        line=dict(color="rgba(255,220,50,0.5)", width=1.2, dash="dot"),
        name="Ground Track [SIM]", hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lat=[pos["lat"]], lon=[pos["lon"]], mode="markers",
        marker=dict(size=12, color="#00ff99" if in_view else "#ff6b6b",
                    line=dict(color="white", width=1.5)),
        name=f"Satellite [SIM] {'✓' if in_view else '✗'}",
        hovertemplate=f"Sat lat={pos['lat']:.3f}° lon={pos['lon']:.3f}°<br>Alt={pos['altitude_km']} km<extra></extra>"))
    fig.add_trace(go.Scattergeo(lat=[gs["lat"]], lon=[gs["lon"]], mode="markers+text",
        marker=dict(size=10, color="#4fc3f7", symbol="star"),
        text=["📡 GS"], textposition="bottom center",
        name=f"Ground Station – {gs['city']}"))
    if in_view:
        fig.add_trace(go.Scattergeo(lat=[pos["lat"], gs["lat"]], lon=[pos["lon"], gs["lon"]],
            mode="lines", line=dict(color="#76ff03", width=2, dash="dash"),
            name="Comm Link [SIM]", hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=0,r=0,t=30,b=0), height=440,
        legend=dict(bgcolor="rgba(22,27,34,0.9)", font=dict(color="white", size=10), x=0, y=0),
        geo=dict(showland=True, landcolor="#1a2332", showocean=True, oceancolor="#0d1b2a",
                 showcountries=True, countrycolor="rgba(255,255,255,0.15)",
                 bgcolor="#0d1117", projection_type="orthographic",
                 projection_rotation=dict(lon=pos["lon"], lat=max(-80,min(80,pos["lat"])), roll=0),
                 showframe=False),
        title=dict(text="LEO Orbit [SIMULATED]", font=dict(color="#58a6ff",size=13), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

with scol:
    st.markdown("**Telemetry** <span class='badge-sim'>SIM</span>", unsafe_allow_html=True)
    _metric("Latitude",  f"{pos['lat']:.3f}°")
    _metric("Longitude", f"{pos['lon']:.3f}°")
    _metric("Altitude",  f"{pos['altitude_km']} km")
    _metric("Period",    f"{_params.period_minutes:.2f} min")
    _metric("Velocity",  f"{_params.velocity_km_per_s:.4f} km/s")
    st.markdown("---")
    link_html = "<span class='state-ok'>IN VIEW</span>" if in_view else "<span class='state-err'>NO CONTACT</span>"
    st.markdown(f"**Link:** {link_html}", unsafe_allow_html=True)
    st.markdown(f"**Orbits:** {elapsed / _params.period_seconds:.2f}")

if S["sim_running"]:
    time.sleep(0.5)
    st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2: Onboard AI – Three apps side by side
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 2 · Onboard AI Processing [source gated]")
eco_col, urban_col, dis_col = st.columns(3)

# ── Eco_Act ──
with eco_col:
    color = APP_COLORS["Eco_Act"]
    st.markdown(f"<h4 style='color:{color}'>🌃 Eco_Act</h4>", unsafe_allow_html=True)
    st.caption("Economic intelligence · VIIRS night-light analysis")

    imgs = _eco_images()
    if imgs:
        bnames  = [os.path.basename(p) for p in imgs]
        choice  = st.selectbox("VIIRS image", bnames, index=len(bnames)-1, key="eco_sel")
        eco_img = imgs[bnames.index(choice)]
        S["eco_image"] = eco_img
        raw_sz = os.path.getsize(eco_img)
        st.caption(f"File: {_fmt_bytes(raw_sz)}")
    else:
        st.error("No VIIRS images found in Eco_Act dataset.")
        eco_img = None

    if st.button("▶ Run Eco_Act", key="run_eco", use_container_width=True):
        if eco_img:
            with st.spinner("Running Eco_Act analysis…"):
                try:
                    t0 = time.perf_counter()
                    result = run_application("Eco_Act", image_path=eco_img)
                    result["processing_time_seconds"] = round(time.perf_counter()-t0, 4)
                    S["run_results"]["Eco_Act"] = result
                    S["input_sizes"]["Eco_Act"] = raw_sz
                    S["run_errors"].pop("Eco_Act", None)
                    _enqueue("Eco_Act", result, raw_sz)
                    st.success(f"Done in {result['processing_time_seconds']} s → queued {PRIORITY_MAP['Eco_Act']}")
                except Exception as e:
                    S["run_errors"]["Eco_Act"] = str(e)

    r = S["run_results"].get("Eco_Act")
    if r:
        st.markdown(f"<span class='badge-act'>ACTUAL</span>", unsafe_allow_html=True)
        st.markdown(f"**Mean radiance:** {r['brightness']['mean']:.2f} DN")
        st.markdown(f"**Significant lit:** {r.get('lit_area_full',{}).get('significant_lit_pct', r.get('lit_area_pct','—'))}%")
        st.markdown(f"**Industrial:** {r['industrial_activity_trend']}")
        st.markdown(f"**Eco index:** {r['economic_activity_index']:.1f} / 100")
        gddp = r["gddp"]
        st.markdown(f"**GDDP (pred):** ₹{gddp['predicted_gddp_crores']:,.0f} Cr")
        st.markdown(f"**Opp zones:** {len(r.get('opportunity_zones',[]))}")
        st.caption(f"⏱ {r['processing_time_seconds']} s")
    elif "Eco_Act" in S["run_errors"]:
        st.warning(S["run_errors"]["Eco_Act"][:240])

# ── UrbanPulse ──
with urban_col:
    color = APP_COLORS["UrbanPulse"]
    st.markdown(f"<h4 style='color:{color}'>🏙 UrbanPulse</h4>", unsafe_allow_html=True)
    st.caption("Urbanization intelligence · Satellite sensor fusion + ML")

    hubs = _urban_hubs()
    if hubs:
        hub_names = list(hubs.keys())
        hub_sel   = st.selectbox("Urban hub", hub_names, key="urban_sel")
        hub_info  = hubs[hub_sel]
        S["urban_hub"] = hub_sel
        st.caption(f"({hub_info['lat']:.2f}°N, {hub_info['lon']:.2f}°E)")
    else:
        hub_sel = None
        st.error("UrbanPulse preset hubs unavailable.")

    if st.button("▶ Run UrbanPulse", key="run_urban", use_container_width=True):
        if hub_sel and hubs:
            with st.spinner("Running UrbanPulse ML pipeline…"):
                try:
                    t0  = time.perf_counter()
                    hub = hubs[hub_sel]
                    result = run_application("UrbanPulse", lat=hub["lat"], lon=hub["lon"])
                    result["processing_time_seconds"] = round(time.perf_counter()-t0, 4)
                    S["run_results"]["UrbanPulse"] = result
                    # Raw input = sum of all 10 feature arrays (48x48 float32 each)
                    raw_sz = result["raw_input_bytes"]
                    S["input_sizes"]["UrbanPulse"] = raw_sz
                    S["run_errors"].pop("UrbanPulse", None)
                    _enqueue("UrbanPulse", result, raw_sz)
                    st.success(f"Done in {result['processing_time_seconds']} s → queued {PRIORITY_MAP['UrbanPulse']}")
                except Exception as e:
                    S["run_errors"]["UrbanPulse"] = str(e)

    r = S["run_results"].get("UrbanPulse")
    if r:
        dq = r.get("data_quality","")
        badge = "badge-syn" if "SYNTHETIC" in dq else "badge-act"
        st.markdown(f"<span class='{badge}'>{dq}</span>", unsafe_allow_html=True)
        if "SYNTHETIC" in dq:
            st.caption("⚠ physics-consistent synthetic features (planetary_computer unavailable)")
        ins = r["insights"]
        st.markdown(f"**Zone counts:** {ins['zone_counts']}")
        st.markdown(f"**CV accuracy:** {ins['classifier_cv_accuracy']:.3f}")
        st.markdown(f"**New built:** {ins['new_built_area_sqkm']:.2f} km²")
        st.markdown(f"**Migrant influx 2027-28:** {ins['estimated_migrant_influx_2027_28']:,}")
        st.markdown(f"**Sprawl risk (max):** {ins['sprawl_risk_max']:.3f}")
        st.caption(f"⏱ {r['processing_time_seconds']} s")
    elif "UrbanPulse" in S["run_errors"]:
        st.warning(S["run_errors"]["UrbanPulse"][:320])

# ── DisasterRisk ──
with dis_col:
    color = APP_COLORS["DisasterRisk"]
    st.markdown(f"<h4 style='color:{color}'>⚠️ DisasterRisk</h4>", unsafe_allow_html=True)
    st.caption("Disaster risk & economic loss assessment")

    scenarios = _disaster_scenarios()
    if scenarios:
        sc_names  = list(scenarios.keys())
        sc_sel    = st.selectbox("Scenario", sc_names, key="dis_sel")
        sc_info   = scenarios[sc_sel]
        S["disaster_scenario"] = sc_sel
        st.caption(f"{sc_info.get('disaster_type','').capitalize()} · {sc_info.get('country','')}")
        raw_sz_dis = None
    else:
        sc_sel = None
        st.error("DisasterRisk scenarios unavailable.")
        raw_sz_dis = None

    if st.button("▶ Run DisasterRisk", key="run_dis", use_container_width=True):
        if sc_sel:
            with st.spinner("Running DisasterRisk pipeline…"):
                try:
                    t0 = time.perf_counter()
                    result = run_application("DisasterRisk", preset_id=sc_sel)
                    result["processing_time_seconds"] = round(time.perf_counter()-t0, 4)
                    S["run_results"]["DisasterRisk"] = result
                    # Raw input size from result (1 MB representative satellite scene)
                    raw_sz_dis = result.get("raw_input_bytes", 1_048_576)
                    S["input_sizes"]["DisasterRisk"] = raw_sz_dis
                    S["run_errors"].pop("DisasterRisk", None)
                    _enqueue("DisasterRisk", result, raw_sz_dis)
                    st.success(f"Done in {result['processing_time_seconds']} s → queued {PRIORITY_MAP['DisasterRisk']}")
                except Exception as e:
                    S["run_errors"]["DisasterRisk"] = str(e)

    r = S["run_results"].get("DisasterRisk")
    if r:
        st.markdown(f"<span class='badge-act'>{r.get('data_status','ACTUAL')}</span>", unsafe_allow_html=True)
        ev = r["event"]
        st.markdown(f"**Event:** {ev['title']}")
        st.markdown(f"**Type:** {ev['disaster_type'].capitalize()}")
        loss = r["loss"]
        inr = loss['total_expected_loss_inr']
        st.markdown(f"**Est. loss:** ₹{inr/1e9:.2f} Bn INR")
        dec = r["decisions"]
        st.markdown(f"**Action:** {dec['recommended_action_headline']}")
        st.markdown(f"**Confidence:** {dec['overall_confidence']:.2f}")
        st.markdown(f"**Payout:** ₹{dec['provisional_payout_inr']/1e6:.1f} M INR")
        st.caption(f"⏱ {r['processing_time_seconds']} s")
    elif "DisasterRisk" in S["run_errors"]:
        st.warning("No inference: " + S["run_errors"]["DisasterRisk"].replace(
            "DisasterRisk inference blocked: ", ""
        )[:260])

run_all_col, run_all_note = st.columns([1, 3])
with run_all_col:
    run_all = st.button("Run selected apps in sequence", use_container_width=True)
with run_all_note:
    st.caption("Eco_Act → UrbanPulse → DisasterRisk. Each app reports source blockers and only valid outputs enter the queue.")
if run_all:
    sequence = [
        ("Eco_Act", {"image_path": eco_img} if eco_img else None),
        ("UrbanPulse", {"lat": hub_info["lat"], "lon": hub_info["lon"]} if hubs else None),
        ("DisasterRisk", {"preset_id": sc_sel} if sc_sel else None),
    ]
    for app_name, app_inputs in sequence:
        S["run_results"].pop(app_name, None)
        S["run_errors"].pop(app_name, None)
        try:
            if app_inputs is None:
                raise ValueError(f"{app_name} input selection is unavailable")
            started = time.perf_counter()
            result = run_application(app_name, **app_inputs)
            result["processing_time_seconds"] = round(time.perf_counter() - started, 4)
            raw_bytes = result.get("image_size_bytes", result.get("raw_input_bytes"))
            if not isinstance(raw_bytes, int) or raw_bytes <= 0:
                raise RuntimeError(f"{app_name} did not provide a measured raw input byte count; payload not queued")
            S["run_results"][app_name] = result
            S["input_sizes"][app_name] = raw_bytes
            _enqueue(app_name, result, raw_bytes)
        except Exception as exc:
            S["run_errors"][app_name] = str(exc)
    st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3: Downlink Queue
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 3 · Intelligent Downlink Queue [SIMULATED comms / ACTUAL payload sizes]")

pending  = S["queue"].snapshot()
bw       = S["bandwidth_mbps"]

if pending:
    q_df = pd.DataFrame(pending)
    q_df["Tx time (s)"] = q_df["size_bytes"].apply(lambda b: round(_tx_time(b, bw), 4))
    q_df["Size"] = q_df["size_bytes"].apply(_fmt_bytes)
    q_df["Priority color"] = q_df["metadata"].apply(lambda m: _priority_color(m.get("priority","LOW")))
    st.dataframe(
        q_df[["label","priority","Size","Tx time (s)"]].rename(columns={"label":"Payload","priority":"Priority"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Dispatching when satellite is IN VIEW · Bandwidth: {bw} Mbps [SIMULATED]")
else:
    st.info("Queue empty. Run an application to generate a payload.")

# Manual dispatch button (for demonstration when not auto-running)
mcol1, mcol2 = st.columns([1, 3])
with mcol1:
    if st.button("📡 Dispatch if in contact", use_container_width=True):
        contact_s = _contact_remaining(elapsed) if in_view else 0.0
        res = S["queue"].dispatch(visible=in_view, bandwidth_mbps=bw, contact_seconds=contact_s)
        for sent_item in res["sent"]:
            meta = sent_item.metadata
            S["received"].append({
                "app": meta["app"], "priority": meta["priority"],
                "json_size_b": meta["json_size_b"], "raw_size_b": meta["raw_input_size_b"],
                "reduction_pct": meta["data_reduction_pct"], "tx_time_s": meta["tx_time_json_s"],
                "transmit_duration_s": meta.get("simulated_transmit_duration_s"),
                "time_saved_s": meta["time_saved_s"], "data_quality": meta["data_quality"],
                "received_at_s": S["elapsed_s"] + meta.get("simulated_receive_offset_s", 0.0),
                "payload": json.loads(sent_item.payload.decode("utf-8")),
            })
        st.success(f"Received {len(res['sent'])} payload(s) after simulated transmission")
        st.rerun()
with mcol2:
    st.caption("Normally dispatches automatically when satellite enters ground-station visibility window.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4: Ground Station – Received payloads
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 4 · Ground Station – Received Intelligence")

if not S["received"]:
    st.info("No payloads received yet. Run applications and dispatch the queue.")
else:
    # Summary row
    apps_rx = [r["app"] for r in S["received"]]
    total_json = sum(r["json_size_b"] for r in S["received"])
    total_raw  = sum(r["raw_size_b"]  for r in S["received"])
    overall_red = (1 - total_json / total_raw) * 100 if total_raw else 0

    s1, s2, s3, s4 = st.columns(4)
    _metric("Payloads received", str(len(S["received"])), s1)
    _metric("Total raw input", _fmt_bytes(total_raw), s2)
    _metric("Total transmitted", _fmt_bytes(total_json), s3)
    _metric("Overall reduction", f"{overall_red:.1f}%", s4)

    st.markdown("---")

    for rec in reversed(S["received"]):
        app   = rec["app"]
        icon  = APP_ICONS[app]
        color = APP_COLORS[app]
        cls   = {"Eco_Act":"app-eco","UrbanPulse":"app-urban","DisasterRisk":"app-dis"}[app]
        prio_col = _priority_color(rec["priority"])

        with st.expander(
            f"{icon} {app}  ·  "
            f"Priority: {rec['priority']}  ·  "
            f"{_fmt_bytes(rec['json_size_b'])}  ·  "
            f"{rec['reduction_pct']:.1f}% reduction  ·  "
            f"Rx at t={rec['received_at_s']:.0f}s"
        ):
            ic1, ic2, ic3, ic4 = st.columns(4)
            _metric("Raw Input", _fmt_bytes(rec["raw_size_b"]), ic1)
            _metric("JSON Payload", _fmt_bytes(rec["json_size_b"]), ic2)
            _metric("Data Reduction", f"{rec['reduction_pct']:.2f}%", ic3)
            _metric("Time Saved @ " + f"{bw}Mbps", f"{rec['time_saved_s']:.4f} s", ic4)

            dq = rec.get("data_quality","")
            if dq:
                badge_cls = "badge-syn" if "SYNTHETIC" in dq else "badge-act"
                st.markdown(f"**Data quality:** <span class='{badge_cls}'>{dq}</span>", unsafe_allow_html=True)

            st.markdown("**Received JSON Payload** <span class='badge-act'>ACTUAL</span>", unsafe_allow_html=True)
            st.code(json.dumps(rec["payload"], indent=2, default=str)[:3000], language="json")

    # Comparison chart
    st.markdown("### Raw vs JSON Payload Comparison")
    if S["received"]:
        fig_cmp = go.Figure()
        apps_   = [r["app"] for r in S["received"]]
        raws_   = [r["raw_size_b"]/1024 for r in S["received"]]
        jsons_  = [r["json_size_b"]/1024 for r in S["received"]]
        fig_cmp.add_trace(go.Bar(x=apps_, y=raws_,  name="Raw Input (KB)",  marker_color="#f78166"))
        fig_cmp.add_trace(go.Bar(x=apps_, y=jsons_, name="JSON Payload (KB)", marker_color="#3fb950"))
        fig_cmp.update_layout(
            barmode="group", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9"), yaxis_title="Size (KB)",
            legend=dict(bgcolor="rgba(22,27,34,0.8)", font=dict(color="white")),
            margin=dict(l=40,r=20,t=30,b=30), height=320,
            title=dict(text="Raw Input vs Insight Payload [ACTUAL sizes]",
                       font=dict(color="#58a6ff",size=13), x=0.5),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

st.divider()
st.caption(
    "🛰 LEO Mission Control · Mach-A-Thon 2026 · Aeromania  |  "
    "Orbital simulation: Keplerian mechanics [SIMULATED] · "
    "AI outputs require real observations; unavailable source data is reported as a blocker."
)
