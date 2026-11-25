"""
Plotly globe and orbit visualisation helpers for the LEO satellite dashboard.
All orbit motion labelled SIMULATED.
"""

import plotly.graph_objects as go
import numpy as np
from typing import List


def build_globe_figure(
    ground_track: List[dict],
    sat_pos: dict,
    gs: dict,
    sat_in_view: bool = False,
    show_comm_link: bool = True,
) -> go.Figure:
    """
    Build an interactive Plotly globe showing:
      - Earth background
      - Satellite ground-track (one full orbit, labelled SIMULATED)
      - Current satellite position (marker)
      - Ground station at Tumkur
      - Communication link line when satellite is in view
    """
    # Ground track line
    track_lats = [p["lat"] for p in ground_track]
    track_lons = [p["lon"] for p in ground_track]

    # Break track at antimeridian crossings (lon jumps > 180°)
    split_lats, split_lons = [], []
    prev_lon = track_lons[0] if track_lons else 0
    for la, lo in zip(track_lats, track_lons):
        if abs(lo - prev_lon) > 180:
            split_lats.append(None)
            split_lons.append(None)
        split_lats.append(la)
        split_lons.append(lo)
        prev_lon = lo

    fig = go.Figure()

    # Ground track (SIMULATED)
    fig.add_trace(go.Scattergeo(
        lat=split_lats,
        lon=split_lons,
        mode="lines",
        line=dict(color="rgba(255,220,50,0.6)", width=1.5, dash="dot"),
        name="🛰 Ground Track (SIMULATED)",
        hoverinfo="skip",
    ))

    # Satellite position (SIMULATED)
    fig.add_trace(go.Scattergeo(
        lat=[sat_pos["lat"]],
        lon=[sat_pos["lon"]],
        mode="markers+text",
        marker=dict(
            size=14,
            color="#00ff99" if sat_in_view else "#ff6b6b",
            symbol="circle",
            line=dict(color="white", width=1.5),
        ),
        text=["🛰 SAT (SIMULATED)"],
        textposition="top center",
        name=f"Satellite @ {sat_pos['lat']:.2f}°, {sat_pos['lon']:.2f}° (SIMULATED)",
        hovertemplate=(
            "<b>Satellite [SIMULATED]</b><br>"
            f"Lat: {sat_pos['lat']:.3f}°<br>"
            f"Lon: {sat_pos['lon']:.3f}°<br>"
            f"Alt: {sat_pos['altitude_km']} km<br>"
            f"In view: {'Yes ✅' if sat_in_view else 'No ❌'}<extra></extra>"
        ),
    ))

    # Ground station
    fig.add_trace(go.Scattergeo(
        lat=[gs["lat"]],
        lon=[gs["lon"]],
        mode="markers+text",
        marker=dict(size=12, color="#4fc3f7", symbol="star", line=dict(color="white", width=1)),
        text=["📡 Tumkur GS"],
        textposition="bottom center",
        name=f"📡 {gs['name']}",
        hovertemplate=(
            f"<b>{gs['name']}</b><br>"
            f"{gs['city']}<br>"
            f"Lat: {gs['lat']}°, Lon: {gs['lon']}°<extra></extra>"
        ),
    ))

    # Communication link when in view
    if show_comm_link and sat_in_view:
        fig.add_trace(go.Scattergeo(
            lat=[sat_pos["lat"], gs["lat"]],
            lon=[sat_pos["lon"], gs["lon"]],
            mode="lines",
            line=dict(color="#76ff03", width=2.5, dash="dash"),
            name="📡 Comm Link (SIMULATED)",
            hoverinfo="skip",
        ))

    # Layout
    fig.update_layout(
        title={
            "text": "🛰 LEO Satellite Orbit Simulation [SIMULATED]",
            "font": {"size": 16, "color": "#ffffff"},
            "x": 0.5,
        },
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        legend=dict(
            bgcolor="rgba(30,40,55,0.85)",
            font=dict(color="white", size=11),
            x=0.01, y=0.01, xanchor="left",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
        geo=dict(
            showland=True,
            landcolor="#1a2332",
            showocean=True,
            oceancolor="#0d1b2a",
            showlakes=True,
            lakecolor="#0d1b2a",
            showcountries=True,
            countrycolor="rgba(255,255,255,0.2)",
            showcoastlines=True,
            coastlinecolor="rgba(255,255,255,0.3)",
            bgcolor="#0d1117",
            projection_type="orthographic",
            projection_rotation=dict(
                lon=sat_pos["lon"],
                lat=max(-80.0, min(80.0, sat_pos["lat"])),
                roll=0,
            ),
            showframe=False,
        ),
    )
    return fig


def build_size_comparison_chart(comm_stats: dict) -> go.Figure:
    """Bar chart comparing raw image vs JSON payload size and transmission times."""
    labels = ["Raw Image", "JSON Insight"]
    sizes_kb = [
        comm_stats["raw_image_size_kb"],
        comm_stats["json_payload_size_kb"],
    ]
    times_s = [
        comm_stats["raw_transmission_time_s"],
        comm_stats["json_transmission_time_s"],
    ]
    colors = ["#ef5350", "#66bb6a"]

    fig = go.Figure()

    # Size bars (primary y-axis)
    fig.add_trace(go.Bar(
        x=labels,
        y=sizes_kb,
        name="Size (KB)",
        marker_color=colors,
        yaxis="y1",
        text=[f"{s:.2f} KB" for s in sizes_kb],
        textposition="outside",
    ))

    # Time bars (secondary y-axis)
    fig.add_trace(go.Bar(
        x=labels,
        y=times_s,
        name="Tx Time (s)",
        marker_color=["#ef9a9a", "#a5d6a7"],
        yaxis="y2",
        text=[f"{t:.4f} s" for t in times_s],
        textposition="outside",
        visible="legendonly",
    ))

    bw = comm_stats["bandwidth_mbps"]
    red = comm_stats["data_reduction_pct"]

    fig.update_layout(
        title=dict(
            text=f"Data Size & Transmission Comparison  |  BW = {bw} Mbps  |  Reduction = {red:.1f}%",
            font=dict(size=14, color="#ffffff"),
            x=0.5,
        ),
        barmode="group",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        yaxis=dict(title="Size (KB)", gridcolor="#30363d"),
        yaxis2=dict(title="Tx Time (s)", overlaying="y", side="right", gridcolor="#30363d"),
        legend=dict(bgcolor="rgba(30,40,55,0.8)", font=dict(color="white")),
        margin=dict(l=50, r=50, t=60, b=40),
        height=380,
    )
    return fig


def build_progress_bar_figure(pct: int, label: str, color: str = "#4fc3f7") -> go.Figure:
    """Mini horizontal bar chart acting as a progress bar."""
    fig = go.Figure(go.Bar(
        x=[pct],
        y=[label],
        orientation="h",
        marker_color=color,
        text=[f"{pct}%"],
        textposition="inside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,
    )
    return fig
