"""
Onboard processor – runs all Eco_Act adapters on a given VIIRS image.
Measures wall-clock processing time and returns a structured metrics dict.
All imports are from the eco_act_adapter package; Eco_Act source is never modified.
"""
import sys
import time
import os
from pathlib import Path
from typing import Dict, Any

# Ensure parent package is importable when running standalone
_LEO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(_LEO_SIM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_LEO_SIM_ROOT.parent))

try:
    from ..eco_act_adapter import brightness, lit_area, industrial, forecast, opportunity
except ImportError:  # standalone execution with LEO_Sim on sys.path
    from eco_act_adapter import brightness, lit_area, industrial, forecast, opportunity


def run(image_path: str) -> Dict[str, Any]:
    """
    Execute the full Eco_Act onboard analysis pipeline on *image_path*.

    Returns a dict with:
      - image metadata (path, filename, size_bytes)
      - brightness stats (mean, max, std, p90, p99)
      - lit_area_pct (% of lit pixels > 15 DN)
      - lit_pixel_count
      - industrial_activity_trend
      - industrial_pixel_ratio
      - economic_activity_index (0-100)
      - gddp (dict: predicted, CI low/high, base, sol_ratio)
      - sum_of_lights
      - opportunity_zones (list of zone dicts, sorted by growth potential)
      - processing_time_seconds
      - status
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    t0 = time.perf_counter()

    b_stats  = brightness.brightness_stats(image_path)
    lit_full = lit_area.lit_area_full_stats(image_path)
    lit_pct  = lit_full["standard_lit_pct"]
    lit_cnt  = lit_area.lit_pixel_count(image_path)
    ind_trend = industrial.trend(image_path)
    ind_ratio = industrial.industrial_pixel_ratio(image_path)
    econ_idx = forecast.economic_index(image_path)
    gddp_data = forecast.gddp(image_path)
    sol = forecast.sum_of_lights(image_path)
    opp_zones = opportunity.zones(image_path)

    elapsed = time.perf_counter() - t0

    return {
        "image_path": image_path,
        "image_filename": os.path.basename(image_path),
        "image_size_bytes": os.path.getsize(image_path),
        "brightness": b_stats,
        "lit_area_pct": round(lit_pct, 3),
        "lit_pixel_count": lit_cnt,
        "lit_area_full": lit_full,
        "industrial_activity_trend": ind_trend,
        "industrial_pixel_ratio": round(ind_ratio, 6),
        "economic_activity_index": econ_idx,
        "gddp": gddp_data,
        "sum_of_lights": round(sol, 1),
        "opportunity_zones": opp_zones,
        "processing_time_seconds": round(elapsed, 4),
        "status": "SUCCESS",
    }
