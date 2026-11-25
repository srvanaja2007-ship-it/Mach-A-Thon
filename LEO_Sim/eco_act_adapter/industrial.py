"""
Adapter for industrial activity analysis.
Mirrors Eco_Act feature_extractor: pixels > high_ind_threshold counted as industrial.
"""
import sys
from pathlib import Path
import numpy as np

_ECO_ACT_SRC = Path(__file__).resolve().parents[2] / "Eco_Act" / "src"
if str(_ECO_ACT_SRC) not in sys.path:
    sys.path.insert(0, str(_ECO_ACT_SRC))

HIGH_IND_THRESHOLD = 70.0  # matches Eco_Act feature_extractor default


def trend(image_path: str, threshold: float = HIGH_IND_THRESHOLD) -> str:
    """Return 'Increasing', 'Moderate', or 'Stable' based on high-intensity pixel ratio."""
    from PIL import Image
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    ratio = float(np.sum(arr > threshold) / arr.size)
    if ratio > 0.015:
        return "Increasing"
    if ratio > 0.005:
        return "Moderate"
    return "Stable"


def high_intensity_pixels(image_path: str, threshold: float = HIGH_IND_THRESHOLD) -> int:
    """Return count of high-intensity (industrial) pixels."""
    from PIL import Image
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return int(np.sum(arr > threshold))


def industrial_pixel_ratio(image_path: str, threshold: float = HIGH_IND_THRESHOLD) -> float:
    """Return ratio of high-intensity pixels to total pixels (0-1)."""
    from PIL import Image
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return float(np.sum(arr > threshold) / arr.size)
