"""
Adapter for opportunity-zone identification.
Mirrors Eco_Act's analyze_opportunity_zones (ai_predictor.py) using a 4x4
spatial grid auto-discovery approach on a single image.
Treats the current image as 'recent' and applies a simple momentum model
(Gaussian diffusion) to estimate a 1-year 'future' raster.
"""
import sys
from pathlib import Path
import numpy as np

_ECO_ACT_SRC = Path(__file__).resolve().parents[2] / "Eco_Act" / "src"
if str(_ECO_ACT_SRC) not in sys.path:
    sys.path.insert(0, str(_ECO_ACT_SRC))

try:
    from ai_predictor import _classify_zone
except ImportError:
    def _classify_zone(zone_type, growth_pct):
        if zone_type == "industrial":
            return "Industrial Corridor Boom", "High Priority: Power grid substations & freight warehouses"
        if zone_type == "core":
            return "Dense Urban Anchor", "Stable Commercial Hub: Branch banking & fintech services"
        if growth_pct > 20.0:
            return "Rapid Growth Hotspot", "High Priority: Infrastructure & power substations"
        if growth_pct > 8.0:
            return "Emerging Growth Zone", "Targeted Investment: Road connectivity & SME parks"
        return "Stable / Low-Growth Zone", "Monitor & Sustain: Micro-grid energy & rural enterprise"


def _simple_future_raster(arr: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    1-year forward raster using Gaussian diffusion (mirrors Eco_Act predict_future_raster_maps).
    """
    from scipy.ndimage import gaussian_filter
    diffused = gaussian_filter(arr, sigma=3.0)
    fringe = ((arr > 12.0) & (arr < 200.0)).astype(np.float32)
    diffusion_gain = diffused * 0.06 * dt * fringe
    np.random.seed(2026)
    noise = np.random.normal(0.0, 0.4, size=arr.shape)
    return np.clip(arr + diffusion_gain + noise, 0.0, 255.0).astype(np.uint8)


def zones(image_path: str) -> list:
    """
    Return a list of opportunity zone dicts using Eco_Act's 4x4 grid discovery.
    Each zone contains name, type, growth_pct, status, recommended_action.
    """
    from PIL import Image
    img = Image.open(image_path).convert("L")
    recent_img = np.array(img, dtype=np.float32)
    future_img = _simple_future_raster(recent_img).astype(np.float32)

    H, W = recent_img.shape
    rows, cols = 4, 4
    row_names = ["North", "North-Central", "South-Central", "South"]
    col_names = ["West", "West-Central", "East-Central", "East"]
    cell_h = max(1, H // rows)
    cell_w = max(1, W // cols)
    zone_list = []

    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < rows - 1 else H
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < cols - 1 else W

            flux_rec = float(np.sum(recent_img[y1:y2, x1:x2]))
            flux_fut = float(np.sum(future_img[y1:y2, x1:x2]))
            mean_rec = float(np.mean(recent_img[y1:y2, x1:x2]))
            growth_pct = ((flux_fut - flux_rec) / max(100.0, flux_rec)) * 100.0

            if mean_rec > 120:
                z_type = "core"
            elif growth_pct > 15:
                z_type = "industrial"
            else:
                z_type = "zone"

            status, action = _classify_zone(z_type, growth_pct)
            zone_list.append({
                "name": f"{row_names[r]}-{col_names[c]}",
                "type": z_type,
                "current_flux": round(flux_rec, 1),
                "future_flux_1yr": round(flux_fut, 1),
                "growth_pct": round(growth_pct, 2),
                "mean_radiance": round(mean_rec, 2),
                "status": status,
                "recommended_action": action,
            })

    # Sort by growth potential
    zone_list.sort(key=lambda z: z["growth_pct"], reverse=True)
    return zone_list
