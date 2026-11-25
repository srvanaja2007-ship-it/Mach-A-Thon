"""
Adapter for brightness extraction.
Directly uses Eco_Act's feature_extractor logic (via numpy/PIL) without
importing a non-existent brightness_extractor module.
"""
import sys
from pathlib import Path
import numpy as np

# Make Eco_Act importable
_ECO_ACT_SRC = Path(__file__).resolve().parents[2] / "Eco_Act" / "src"
if str(_ECO_ACT_SRC) not in sys.path:
    sys.path.insert(0, str(_ECO_ACT_SRC))


def mean_brightness(image_path: str) -> float:
    """Return the mean night-light radiance (0-255) for a VIIRS image."""
    from PIL import Image
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return float(arr.mean())


def max_brightness(image_path: str) -> float:
    """Return the maximum pixel radiance."""
    from PIL import Image
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return float(arr.max())


def brightness_stats(image_path: str) -> dict:
    """Return a dict with mean, max, std of radiance."""
    from PIL import Image
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }
