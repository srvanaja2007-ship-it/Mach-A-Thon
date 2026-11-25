"""
planner_engine.py — AI Corridor Auto-Router & Rupee Savings Simulator
UrbanPulse AI | Aeromania '26 Mach-A-Thon | Track A: Migration & Urbanization

Automatically routes shortest, cheapest bus/power corridors from Night-Light
Gap settlements to nearest OSM arterial roads using NetworkX Dijkstra,
then calculates "Act Now vs. Wait 3 Years" policy ROI in Indian Rupees.
"""

from __future__ import annotations

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

GRID_SIZE_M = 250  # metres per cell
N_CELLS     = 48

# ---------------------------------------------------------------------------
# Cost parameters (backed by Karnataka PWD & BBMP corridor costing reports)
# ---------------------------------------------------------------------------
# Bus-road corridor cost: ₹ 2.5 Cr / km (paved 2-lane village road)
# Power-line corridor cost: ₹ 0.8 Cr / km (11 kV rural distribution line)
COST_ROAD_CR_PER_KM   = 2.5
COST_POWER_CR_PER_KM  = 0.8
TOTAL_INFRA_CR_PER_KM = COST_ROAD_CR_PER_KM + COST_POWER_CR_PER_KM   # 3.3 Cr/km

# Retrofit cost multiplier if deferred 3 years (land appreciation + demolition)
RETROFIT_MULTIPLIER = 3.4

# Households per sq-km in informal peri-urban Karnataka (Census + BBMP data)
HH_PER_SQKM = 1_200

# INR → USD conversion
INR_TO_USD  = 83.5    # 1 USD = 83.5 INR
CR_TO_INR   = 1e7     # 1 Crore = 10M INR
USD_MIL     = 1e6

# ---------------------------------------------------------------------------
# Build weighted corridor graph over 48 × 48 grid
# ---------------------------------------------------------------------------
def _build_grid_graph(zone_preds: np.ndarray,
                      water_mask: np.ndarray,
                      road_dist_m: np.ndarray) -> nx.Graph:
    """
    Build a weighted NetworkX grid graph where:
      - Each node = one 250m cell (row, col)
      - Edge weight = movement cost (high for water/green, low for open land)
    """
    G = nx.grid_2d_graph(N_CELLS, N_CELLS)

    for r in range(N_CELLS):
        for c in range(N_CELLS):
            # Base cost per edge = 1 unit (250 m)
            cost = 1.0
            # Penalty zones: water/green → very expensive to build through
            if water_mask[r, c] > 0.5:
                cost = 50.0   # effectively impassable
            elif zone_preds[r, c] == 3:  # green buffer
                cost = 20.0
            elif zone_preds[r, c] == 0:  # tech core (already built)
                cost = 5.0
            G.nodes[(r, c)]["cost"]      = cost
            G.nodes[(r, c)]["zone"]      = int(zone_preds[r, c])
            G.nodes[(r, c)]["road_dist"] = float(road_dist_m[r, c])

    # Apply node cost to edges
    for u, v in G.edges():
        edge_cost = (G.nodes[u]["cost"] + G.nodes[v]["cost"]) / 2.0
        G[u][v]["weight"] = edge_cost

    return G


def _cell_to_latlon(row: int, col: int,
                    lat: float, lon: float) -> tuple[float, float]:
    """Convert grid (row, col) → (lat, lon)."""
    half = (N_CELLS / 2 - 0.5) * GRID_SIZE_M / 1000  # km from centre
    cell_lat = lat - half / 111.0 + row * GRID_SIZE_M / 1000 / 111.0
    cell_lon = (lon - half / (111.0 * np.cos(np.radians(lat)))
                + col * GRID_SIZE_M / 1000 / (111.0 * np.cos(np.radians(lat))))
    return float(cell_lat), float(cell_lon)


# ---------------------------------------------------------------------------
# Find top Night-Light Gap clusters → route corridors
# ---------------------------------------------------------------------------
def compute_corridors(ml_result: dict, top_n: int = 5) -> list[dict]:
    """
    For the top-N Night-Light Gap clusters, compute:
      - Shortest Dijkstra path to nearest existing road
      - Corridor length (km)
      - GeoJSON LineString coordinates
      - Households served
      - Cost estimate (₹ Crore)
    """
    zone_preds  = ml_result["zone_preds"]
    water_mask  = ml_result["water_mask"]
    road_dist_m = ml_result["road_dist_m"]
    lat         = ml_result["lat"]
    lon         = ml_result["lon"]
    sprawl_prob = ml_result["sprawl_prob"]

    G = _build_grid_graph(zone_preds, water_mask, road_dist_m)

    # Find Night-Light Gap cells (zone 2) sorted by sprawl probability
    nlg_cells = [
        (r, c)
        for r in range(N_CELLS) for c in range(N_CELLS)
        if zone_preds[r, c] == 2
    ]
    if not nlg_cells:
        log.warning("No Night-Light Gap cells found — using high-risk cells.")
        # Fall back to high sprawl-risk cells
        flat_sprawl = sprawl_prob.ravel()
        top_idx = np.argsort(flat_sprawl)[::-1][:top_n * 4]
        nlg_cells = [(i // N_CELLS, i % N_CELLS) for i in top_idx]

    # Sort NLG cells by sprawl probability (descending)
    nlg_cells.sort(key=lambda rc: -sprawl_prob[rc[0], rc[1]])

    # Find cells that are already well-connected (road_dist < 300m) → these are targets
    road_nodes = [
        (r, c)
        for r in range(N_CELLS) for c in range(N_CELLS)
        if road_dist_m[r, c] < 300
    ]
    if not road_nodes:
        # Pick the row/col closest to centre as road proxy
        road_nodes = [(N_CELLS // 2, N_CELLS // 2)]

    # Cluster NLG cells spatially (avoid routing duplicate corridors)
    corridors: list[dict] = []
    used_sources: set = set()

    # Simple greedy spatial clustering: pick top NLG cell, skip neighbours within 5 cells
    for nlg_cell in nlg_cells:
        if len(corridors) >= top_n:
            break
        # Check if this cell is too close to already-picked sources
        too_close = any(
            abs(nlg_cell[0] - s[0]) < 5 and abs(nlg_cell[1] - s[1]) < 5
            for s in used_sources
        )
        if too_close:
            continue
        used_sources.add(nlg_cell)

        # Find nearest road node (Dijkstra single-source to all road nodes)
        try:
            # Use shortest path to nearest road node
            best_path = None
            best_len  = float("inf")
            # Limit road_nodes to nearest 20 for speed
            rn_sorted = sorted(road_nodes,
                               key=lambda rc: abs(rc[0] - nlg_cell[0]) + abs(rc[1] - nlg_cell[1]))[:20]
            for rn in rn_sorted:
                if rn == nlg_cell:
                    continue
                try:
                    path_len = nx.dijkstra_path_length(G, nlg_cell, rn, weight="weight")
                    if path_len < best_len:
                        best_len  = path_len
                        best_path = nx.dijkstra_path(G, nlg_cell, rn, weight="weight")
                except nx.NetworkXNoPath:
                    continue
        except Exception as e:
            log.warning("Path finding failed: %s", e)
            continue

        if best_path is None or len(best_path) < 2:
            continue

        # Convert path cells to lat/lon coords
        coords = [
            list(_cell_to_latlon(r, c, lat, lon))
            for r, c in best_path
        ]

        # Physical corridor length (km)
        corridor_km = len(best_path) * GRID_SIZE_M / 1000

        # Served households (cells within 1 km of corridor with zone==2)
        served_cells = sum(
            1 for r2 in range(N_CELLS) for c2 in range(N_CELLS)
            if zone_preds[r2, c2] == 2 and
               min(abs(r2 - p[0]) + abs(c2 - p[1]) for p in best_path) <= 4
        )
        served_hh = max(50, served_cells * HH_PER_SQKM * (GRID_SIZE_M / 1000) ** 2)

        # Cost estimate
        cost_cr = corridor_km * TOTAL_INFRA_CR_PER_KM

        # Sprawl risk of source cell
        risk_score = float(sprawl_prob[nlg_cell[0], nlg_cell[1]])

        # Source cell coordinates
        src_lat, src_lon = _cell_to_latlon(nlg_cell[0], nlg_cell[1], lat, lon)
        tgt_lat, tgt_lon = coords[-1]

        corridors.append({
            "id":           len(corridors) + 1,
            "source_cell":  nlg_cell,
            "source_lat":   src_lat,
            "source_lon":   src_lon,
            "target_lat":   tgt_lat,
            "target_lon":   tgt_lon,
            "corridor_km":  round(corridor_km, 2),
            "served_hh":    int(served_hh),
            "cost_cr":      round(cost_cr, 2),
            "risk_score":   round(risk_score, 3),
            "geojson": {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[c[1], c[0]] for c in coords],  # GeoJSON: [lon, lat]
                },
                "properties": {
                    "corridor_id":  len(corridors) + 1,
                    "corridor_km":  round(corridor_km, 2),
                    "served_hh":    int(served_hh),
                    "cost_cr":      round(cost_cr, 2),
                    "risk_score":   round(risk_score, 3),
                },
            },
        })

    # Sort by risk score (highest first)
    corridors.sort(key=lambda c: -c["risk_score"])
    for i, c in enumerate(corridors):
        c["rank"] = i + 1

    return corridors


# ---------------------------------------------------------------------------
# "Act Now vs. Wait 3 Years" ROI Simulator
# ---------------------------------------------------------------------------
def simulate_roi(ml_result: dict,
                 corridors: list[dict],
                 early_budget_cr: float) -> dict:
    """
    Simulate the municipal savings from early corridor reservation vs.
    waiting 3 years (demolition + retrofit).

    Returns a dict of all KPIs for the frontend dashboard.
    """
    if not corridors:
        return _empty_roi(early_budget_cr)

    # Total natural cost of all corridors if built early
    total_natural_cost_cr = sum(c["cost_cr"] for c in corridors)

    # Corridor coverage ratio with given budget
    coverage_ratio = min(1.0, early_budget_cr / total_natural_cost_cr) if total_natural_cost_cr > 0 else 0.0

    # Select corridors that fit within budget (greedy by cost)
    funded_corridors = []
    running_cost = 0.0
    for c in sorted(corridors, key=lambda x: x["cost_cr"]):
        if running_cost + c["cost_cr"] <= early_budget_cr:
            funded_corridors.append(c)
            running_cost += c["cost_cr"]

    if not funded_corridors:
        funded_corridors = corridors[:1]
        running_cost = funded_corridors[0]["cost_cr"]

    # Sprawl prevented
    n_nlg = ml_result["n_nlg_alerts"]
    nlg_area_sqkm = ml_result["nlg_area_sqkm"]
    sprawl_prevented_sqkm = nlg_area_sqkm * coverage_ratio * 0.72   # 72% effectiveness

    # New migrant pop influx covered
    influx = ml_result["influx_2027_28"]
    influx_covered = int(influx * coverage_ratio * 0.65)

    # Households connected
    hh_connected = sum(c["served_hh"] for c in funded_corridors)

    # Cost if deferred 3 years (encroachment → retrofit)
    deferred_cost_cr = running_cost * RETROFIT_MULTIPLIER

    # Net savings
    net_savings_cr  = round(deferred_cost_cr - running_cost, 2)
    net_savings_usd_m = round(net_savings_cr * CR_TO_INR / INR_TO_USD / USD_MIL, 2)

    # ROI ratio
    roi_ratio = round(deferred_cost_cr / running_cost, 2) if running_cost > 0 else 0.0

    return {
        "early_budget_cr":       round(early_budget_cr, 1),
        "funded_corridors":      funded_corridors,
        "n_funded":              len(funded_corridors),
        "total_funded_cost_cr":  round(running_cost, 2),
        "deferred_cost_cr":      round(deferred_cost_cr, 2),
        "net_savings_cr":        net_savings_cr,
        "net_savings_usd_m":     net_savings_usd_m,
        "roi_ratio":             roi_ratio,
        "sprawl_prevented_sqkm": round(sprawl_prevented_sqkm, 2),
        "influx_covered":        influx_covered,
        "hh_connected":          hh_connected,
        "coverage_ratio":        round(coverage_ratio * 100, 1),
    }


def _empty_roi(budget: float) -> dict:
    return {
        "early_budget_cr": budget, "funded_corridors": [],
        "n_funded": 0, "total_funded_cost_cr": 0,
        "deferred_cost_cr": 0, "net_savings_cr": 0,
        "net_savings_usd_m": 0, "roi_ratio": 0,
        "sprawl_prevented_sqkm": 0, "influx_covered": 0,
        "hh_connected": 0, "coverage_ratio": 0,
    }


# ---------------------------------------------------------------------------
# Full planner pipeline
# ---------------------------------------------------------------------------
def run_planner(ml_result: dict, early_budget_cr: float = 50.0) -> dict:
    """
    Given ML result dict, compute corridors + ROI.
    Returns extended dict with corridors and ROI stats.
    """
    lat = ml_result["lat"]
    lon = ml_result["lon"]
    key = hashlib.md5(f"{lat:.4f}_{lon:.4f}".encode()).hexdigest()[:12]
    plan_cache = CACHE_DIR / f"{key}_plan.pkl"

    if plan_cache.exists():
        log.info("Loading planner from cache: %s", plan_cache)
        with open(plan_cache, "rb") as f:
            corridors = pickle.load(f)
    else:
        log.info("Computing AI corridors …")
        corridors = compute_corridors(ml_result, top_n=5)
        with open(plan_cache, "wb") as f:
            pickle.dump(corridors, f, protocol=4)
        log.info("Planner cached → %s", plan_cache)

    roi = simulate_roi(ml_result, corridors, early_budget_cr)

    return {
        **ml_result,
        "corridors":  corridors,
        "roi":        roi,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_pipeline import PRESET_HUBS, fetch_location_data
    from ml_engine import run_ml_pipeline

    hub = list(PRESET_HUBS.items())[0]
    name, info = hub
    print(f"\n=== {name} ===")
    data    = fetch_location_data(info["lat"], info["lon"])
    ml_res  = run_ml_pipeline(data)
    plan    = run_planner(ml_res, early_budget_cr=50.0)

    print(f"  Corridors found : {len(plan['corridors'])}")
    for c in plan["corridors"]:
        print(f"  Rank {c['rank']}: {c['corridor_km']:.1f} km, "
              f"{c['served_hh']:,} HH, ₹{c['cost_cr']:.1f} Cr  risk={c['risk_score']:.3f}")
    roi = plan["roi"]
    print(f"\n  Net Savings : ₹{roi['net_savings_cr']:.1f} Cr  "
          f"(${roi['net_savings_usd_m']:.1f} M)")
    print(f"  HH Connected: {roi['hh_connected']:,}")

