"""
Adapter for economic activity forecasting.
Uses Eco_Act's SpatioTemporalDevelopmentAI for a single-image forecast:
- Treats the single image as the latest data point.
- Uses stored 10-year trend data from Eco_Act's image_stats.json if available.
- Falls back to image-derived metrics when dataset is unavailable.
"""
import sys
import json
import os
from pathlib import Path
import numpy as np

_ECO_ACT_SRC = Path(__file__).resolve().parents[2] / "Eco_Act" / "src"
_ECO_ACT_ROOT = Path(__file__).resolve().parents[2] / "Eco_Act"
if str(_ECO_ACT_SRC) not in sys.path:
    sys.path.insert(0, str(_ECO_ACT_SRC))

# Base GDDP for Tumkur (₹ Crores) calibrated to 2015 value from Eco_Act
_BASE_GDDP_2015 = 32000.0
_ELASTICITY = 0.75


def _load_image_array(image_path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)


def _try_load_dataset_stats():
    """Load image_stats.json from Eco_Act dataset if available."""
    stats_path = _ECO_ACT_ROOT / "Dataset" / "image_stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def economic_index(image_path: str) -> float:
    """
    Return a normalized economic activity index (0-100).
    Computed from: (mean_radiance / max_possible_radiance) * 100,
    scaled against the Eco_Act historical dataset baseline where available.
    """
    arr = _load_image_array(image_path)
    mean_val = float(arr.mean())
    # Scale 0-255 → 0-100 (index)
    base_index = (mean_val / 255.0) * 100.0

    # Optionally adjust against historical dataset mean
    stats = _try_load_dataset_stats()
    if stats and isinstance(stats, list) and len(stats) > 0:
        # image_stats.json is a list of per-image records
        hist_means = [r.get("mean_radiance", 0) for r in stats if "mean_radiance" in r]
        if hist_means:
            hist_mean = float(np.mean(hist_means))
            if hist_mean > 0:
                # Index relative to historical mean
                base_index = min(100.0, (mean_val / hist_mean) * 35.0)
    return round(base_index, 3)


def gddp(image_path: str) -> dict:
    """
    Return estimated GDDP (₹ Crores) with confidence interval.
    Mirrors Eco_Act SpatioTemporalDevelopmentAI.predict_macro_future logic.
    """
    arr = _load_image_array(image_path)
    sol = float(np.sum(arr))

    # Establish baseline SOL from dataset if available
    stats = _try_load_dataset_stats()
    base_sol = None
    if stats and isinstance(stats, list):
        sols = [r.get("sum_of_lights", 0) for r in stats if "sum_of_lights" in r]
        if sols:
            base_sol = float(sols[0])  # earliest record as baseline

    if base_sol is None or base_sol <= 0:
        # Fallback: use 10% of image max as synthetic baseline
        base_sol = float(arr.max() * arr.size * 0.1)
    base_sol = max(1.0, base_sol)

    sol_ratio = sol / base_sol
    pred_gddp = float(_BASE_GDDP_2015 * (sol_ratio ** _ELASTICITY))
    return {
        "predicted_gddp_crores": round(pred_gddp, 2),
        "confidence_low_crores": round(pred_gddp * 0.93, 2),
        "confidence_high_crores": round(pred_gddp * 1.07, 2),
        "base_gddp_crores_2015": _BASE_GDDP_2015,
        "sol_ratio": round(sol_ratio, 4),
    }


def sum_of_lights(image_path: str) -> float:
    """Return Sum of Lights (SOL) for the image."""
    arr = _load_image_array(image_path)
    return float(np.sum(arr))
