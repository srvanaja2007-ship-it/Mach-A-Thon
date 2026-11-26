"""
api_server.py — Lightweight Starlette API & Static Host for UrbanPulse AI React UI
Aeromania '26 Mach-A-Thon | Track A: Migration & Urbanization
Imports existing data_pipeline, ml_engine, and planner_engine without modifying core logic.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from data_pipeline import BORDER_SECTORS, PRESET_HUBS, fetch_location_data
from ml_engine import ZONE_COLORS, ZONE_LABELS, ZONE_SHORT, run_ml_pipeline
from planner_engine import run_planner, simulate_roi

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("urbanpulse_api")

# In-memory cache of base plans keyed by (round(lat,4), round(lon,4))
_PLAN_CACHE: dict[tuple[float, float], dict[str, Any]] = {}


def _get_base_plan(lat: float, lon: float) -> dict[str, Any]:
    key = (round(lat, 4), round(lon, 4))
    if key not in _PLAN_CACHE:
        data = fetch_location_data(lat, lon)
        ml_res = run_ml_pipeline(data)
        _PLAN_CACHE[key] = run_planner(ml_res, early_budget_cr=50.0)
    return _PLAN_CACHE[key]


def _serialize_plan(plan: dict[str, Any], roi: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy grid arrays + planner output into compact JSON for React + Leaflet."""
    n = int(plan["n_cells"])
    zone_preds = plan["zone_preds"]
    sprawl_prob = plan["sprawl_prob"]
    grid_lat = plan["grid_lat"]
    grid_lon = plan["grid_lon"]
    delta_ndbi = plan["delta_ndbi"]
    viirs_dnb = plan["viirs_dnb"]
    vv_db = plan["vv_db"]
    road_dist_m = plan["road_dist_m"]

    # Full 48x48 (2,304 cells) seamless grid for organic map overlay
    cells = []
    for r in range(n):
        for c in range(n):
            cells.append({
                "lat": round(float(grid_lat[r, c]), 5),
                "lon": round(float(grid_lon[r, c]), 5),
                "zone": int(zone_preds[r, c]),
                "sprawl": round(float(sprawl_prob[r, c]), 3),
                "delta_ndbi": round(float(delta_ndbi[r, c]), 4),
                "viirs": round(float(viirs_dnb[r, c]), 2),
                "vv_db": round(float(vv_db[r, c]), 2),
                "road_m": round(float(road_dist_m[r, c]), 1),
            })

    # Full-resolution Night-Light Gap cells (Zone 2, capped to top 160 for fast Leaflet render)
    nlg_mask = zone_preds == 2
    nlg_indices = np.argwhere(nlg_mask)
    nlg_cells = []
    for r, c in nlg_indices:
        nlg_cells.append({
            "lat": round(float(grid_lat[r, c]), 5),
            "lon": round(float(grid_lon[r, c]), 5),
            "sprawl": round(float(sprawl_prob[r, c]), 3),
            "delta_ndbi": round(float(delta_ndbi[r, c]), 4),
            "viirs": round(float(viirs_dnb[r, c]), 2),
            "vv_db": round(float(vv_db[r, c]), 2),
            "road_m": round(float(road_dist_m[r, c]), 1),
        })
    nlg_cells.sort(key=lambda x: -x["sprawl"])
    nlg_cells = nlg_cells[:160]

    # Radar VV distribution summary per zone for chart
    radar_stats = []
    for z in range(4):
        vals = vv_db[zone_preds == z]
        if len(vals) > 0:
            radar_stats.append({
                "zone": z,
                "label": ZONE_SHORT[z],
                "color": ZONE_COLORS[z],
                "count": int(len(vals)),
                "p10": round(float(np.percentile(vals, 10)), 2),
                "mean": round(float(np.mean(vals)), 2),
                "p90": round(float(np.percentile(vals, 90)), 2),
            })

    # Feature importances sorted descending
    feat_imp = [
        {"feature": k, "importance": round(float(v), 4)}
        for k, v in sorted(plan.get("feat_importances", {}).items(), key=lambda kv: -kv[1])
    ]

    # Corridors (strip heavy non-JSON if any)
    corridors = []
    for c in plan.get("corridors", []):
        corridors.append({
            "rank": int(c["rank"]),
            "source_lat": round(float(c["source_lat"]), 5),
            "source_lon": round(float(c["source_lon"]), 5),
            "target_lat": round(float(c["target_lat"]), 5),
            "target_lon": round(float(c["target_lon"]), 5),
            "corridor_km": round(float(c["corridor_km"]), 2),
            "served_hh": int(c["served_hh"]),
            "cost_cr": round(float(c["cost_cr"]), 2),
            "risk_score": round(float(c["risk_score"]), 3),
            "coords": c.get("coords", []),
        })

    return {
        "lat": float(plan["lat"]),
        "lon": float(plan["lon"]),
        "n_cells": n,
        "grid_size_m": int(plan.get("grid_size_m", 250)),
        "sources": plan.get("sources", {}),
        "cv_accuracy": round(float(plan.get("cv_accuracy", 0.0)), 4),
        "new_built_sqkm": round(float(plan.get("new_built_sqkm", 0.0)), 2),
        "n_nlg_alerts": int(plan.get("n_nlg_alerts", 0)),
        "nlg_area_sqkm": round(float(plan.get("nlg_area_sqkm", 0.0)), 2),
        "influx_2027_28": int(plan.get("influx_2027_28", 0)),
        "sprawl_prob_mean": round(float(plan.get("sprawl_prob_mean", 0.0)), 3),
        "sprawl_prob_max": round(float(plan.get("sprawl_prob_max", 0.0)), 3),
        "n_zones": {str(k): int(v) for k, v in plan.get("n_zones", {}).items()},
        "zone_labels": {str(k): v for k, v in ZONE_LABELS.items()},
        "zone_short": {str(k): v for k, v in ZONE_SHORT.items()},
        "zone_colors": {str(k): v for k, v in ZONE_COLORS.items()},
        "roi": roi,
        "corridors": corridors,
        "cells": cells,
        "nlg_cells": nlg_cells,
        "radar_stats": radar_stats,
        "feat_importances": feat_imp,
    }


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "UrbanPulse AI"})


async def get_presets(request: Request) -> JSONResponse:
    return JSONResponse({"presets": PRESET_HUBS, "border_sectors": BORDER_SECTORS})


async def analyze(request: Request) -> JSONResponse:
    lat = float(request.query_params.get("lat", 13.20))
    lon = float(request.query_params.get("lon", 77.70))
    budget_cr = float(request.query_params.get("budget_cr", 50.0))

    base_plan = await asyncio.to_thread(_get_base_plan, lat, lon)
    roi = simulate_roi(base_plan, base_plan["corridors"], early_budget_cr=budget_cr)
    return JSONResponse(_serialize_plan(base_plan, roi))


async def update_roi(request: Request) -> JSONResponse:
    lat = float(request.query_params.get("lat", 13.20))
    lon = float(request.query_params.get("lon", 77.70))
    budget_cr = float(request.query_params.get("budget_cr", 50.0))

    base_plan = await asyncio.to_thread(_get_base_plan, lat, lon)
    roi = simulate_roi(base_plan, base_plan["corridors"], early_budget_cr=budget_cr)
    return JSONResponse({"roi": roi})


async def geocode(request: Request) -> JSONResponse:
    q = request.query_params.get("q", "").strip()
    if not q:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    parts = q.replace(" ", "").split(",")
    if len(parts) == 2:
        try:
            return JSONResponse({"lat": float(parts[0]), "lon": float(parts[1]), "label": q})
        except ValueError:
            pass

    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="urbanpulse_ai_react_v1")
        loc = geolocator.geocode(q + ", India", timeout=10)
        if loc:
            return JSONResponse({"lat": loc.latitude, "lon": loc.longitude, "label": loc.address})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"error": "Location not found"}, status_code=404)


routes: list[Any] = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/presets", get_presets, methods=["GET"]),
    Route("/api/analyze", analyze, methods=["GET"]),
    Route("/api/roi", update_roi, methods=["GET"]),
    Route("/api/geocode", geocode, methods=["GET"]),
]

DIST_DIR = PROJECT_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    routes.append(Mount("/", app=StaticFiles(directory=str(DIST_DIR), html=True), name="static"))

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="info")
