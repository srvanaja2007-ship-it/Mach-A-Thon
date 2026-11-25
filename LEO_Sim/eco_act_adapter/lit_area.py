"""
Adapter for lit-area extraction.

IMPORTANT – about the Eco_Act VIIRS PNGs:
  These are preprocessed monthly composite PNGs (1024×1024, grayscale).
  The minimum pixel value is ~51 DN (not 0), because the composites have already
  had dark ocean / background normalised away. No pixel falls below 31 DN.
  Therefore the Eco_Act default threshold of 15 DN gives 100% lit for every image,
  which is meaningless. This adapter uses TWO thresholds:
    - standard_lit_threshold = 15 DN  (kept for compatibility, but correctly flags
      that all pixels exceed it in these normalised composites)
    - significant_lit_threshold = adaptive, computed as the per-image p50 (median)
      from image_stats.json when available; otherwise 85th-percentile of the image.
      Pixels above this threshold represent meaningfully lit / active areas.

We report BOTH and clearly label them in the payload.
"""
import sys
import os
import json
from pathlib import Path
import numpy as np

_ECO_ACT_SRC  = Path(__file__).resolve().parents[2] / "Eco_Act" / "src"
_ECO_ACT_ROOT = Path(__file__).resolve().parents[2] / "Eco_Act"
if str(_ECO_ACT_SRC) not in sys.path:
    sys.path.insert(0, str(_ECO_ACT_SRC))

STANDARD_LIT_THRESHOLD = 15.0   # original Eco_Act value


def _load_arr(image_path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)


def _stats_record(image_filename: str):
    """Look up pre-computed stats for this image in image_stats.json."""
    stats_path = _ECO_ACT_ROOT / "Dataset" / "image_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    fname = os.path.basename(image_filename)
    for rec in stats:
        if rec.get("file") == fname:
            return rec
    return None


def _significant_threshold(arr: np.ndarray, stats_rec) -> float:
    """
    Return a threshold above which pixels are 'significantly lit'.
    Uses the dataset's p50 (median DN) for this image if available;
    otherwise falls back to the image's own 85th percentile.
    """
    if stats_rec and "p50" in stats_rec:
        return float(stats_rec["p50"])
    return float(np.percentile(arr, 85))


def percentage(image_path: str, threshold: float = STANDARD_LIT_THRESHOLD) -> float:
    """
    Return percentage of pixels above *threshold*.
    With these normalised PNGs: threshold ≤ 30 → always 100%.
    Use lit_area_full_stats() for a meaningful breakdown.
    """
    arr = _load_arr(image_path)
    lit = int(np.sum(arr > threshold))
    return float((lit / arr.size) * 100.0)


def lit_pixel_count(image_path: str, threshold: float = STANDARD_LIT_THRESHOLD) -> int:
    arr = _load_arr(image_path)
    return int(np.sum(arr > threshold))


def lit_area_full_stats(image_path: str) -> dict:
    """
    Return a comprehensive lit-area breakdown:
      - standard_lit_pct: pixels > 15 DN (always 100% for these composites – noted)
      - significant_lit_pct: pixels above per-image median (meaningful signal)
      - high_intensity_pct: pixels > p95 of this image (hotspots only)
      - image_min / image_max / image_mean: actual pixel statistics
      - threshold_note: explains why standard threshold yields 100%
    """
    arr = _load_arr(image_path)
    stats_rec = _stats_record(image_path)
    total = arr.size

    standard_lit = int(np.sum(arr > STANDARD_LIT_THRESHOLD))
    sig_thresh    = _significant_threshold(arr, stats_rec)
    sig_lit       = int(np.sum(arr > sig_thresh))
    p95           = float(stats_rec["p95"]) if stats_rec and "p95" in stats_rec else float(np.percentile(arr, 95))
    high_lit      = int(np.sum(arr > p95))

    return {
        "standard_lit_pct":    round((standard_lit / total) * 100.0, 2),
        "standard_threshold":  STANDARD_LIT_THRESHOLD,
        "significant_lit_pct": round((sig_lit / total) * 100.0, 2),
        "significant_threshold": round(sig_thresh, 1),
        "high_intensity_pct":  round((high_lit / total) * 100.0, 2),
        "high_intensity_threshold": round(p95, 1),
        "image_min":  round(float(arr.min()), 1),
        "image_max":  round(float(arr.max()), 1),
        "image_mean": round(float(arr.mean()), 3),
        "threshold_note": (
            "Standard Eco_Act threshold (15 DN) yields 100% because these PNG "
            "composites are pre-normalised: background pixels start at ~51 DN. "
            "significant_lit_pct uses the per-image median as threshold (meaningful signal). "
            "high_intensity_pct marks saturated/urban core pixels."
        ),
    }
