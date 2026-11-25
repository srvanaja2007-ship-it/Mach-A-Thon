import numpy as np
import cv2
import pandas as pd
from download_online_dataset import LANDMARKS, coord_to_pixel

def extract_landmark_radiance(img, w=1024, h=1024, radius=24):
    """
    Extracts integrated radiant flux around each key economic landmark.
    """
    results = {}
    for lm in LANDMARKS:
        px, py = coord_to_pixel(lm['lat'], lm['lon'], w=w, h=h)
        x1 = max(0, px - radius)
        x2 = min(w, px + radius + 1)
        y1 = max(0, py - radius)
        y2 = min(h, py + radius + 1)

        patch = img[y1:y2, x1:x2]
        flux = float(np.sum(patch))
        mean_flux = float(np.mean(patch))
        key = lm['name'].lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        results[f"flux_{key}"] = flux
        results[f"mean_{key}"] = mean_flux
    return results

def extract_geospatial_features(images, timestamps, years, months, lit_threshold=15.0, high_ind_threshold=70.0):
    """
    Extracts multi-scale geospatial radiance and development metrics for all time steps.
    """
    N, H, W = images.shape
    records = []
    yy, xx = np.mgrid[0:H, 0:W]

    print(f"[FeatureExtractor] Extracting spatial & radiant flux dynamics across {N} frames ({H}x{W})...")

    for i in range(N):
        img = images[i].astype(np.float32)
        t_dec = timestamps[i]
        yr = years[i]
        mo = months[i]

        # 1. Macro Radiance Metrics
        sol = float(np.sum(img))
        mean_rad = float(np.mean(img))
        max_rad = float(np.max(img))

        # 2. Lit Area and Industrial Hotspots
        lit_mask = (img > lit_threshold).astype(np.uint8)
        lit_pixels = int(np.sum(lit_mask))
        lit_area_pct = float((lit_pixels / (H * W)) * 100.0)

        high_ind_mask = (img > high_ind_threshold).astype(np.uint8)
        high_ind_pixels = int(np.sum(high_ind_mask))

        # 3. Center of Mass (Spatial Trajectory / Expansion Drift)
        total_w = np.sum(img) + 1e-6
        com_x = float(np.sum(xx * img) / total_w)
        com_y = float(np.sum(yy * img) / total_w)

        # 4. Landmark-Specific Flux
        lm_flux = extract_landmark_radiance(img, w=W, h=H)

        row = {
            'timestamp': t_dec,
            'year': yr,
            'month': mo,
            'sum_of_lights': sol,
            'mean_radiance': mean_rad,
            'max_radiance': max_rad,
            'lit_area_pixels': lit_pixels,
            'lit_area_pct': lit_area_pct,
            'high_intensity_industrial_pixels': high_ind_pixels,
            'center_of_mass_x': com_x,
            'center_of_mass_y': com_y,
        }
        row.update(lm_flux)
        records.append(row)

    df = pd.DataFrame(records)

    # 12-Month Moving Average to smooth out monsoon seasonality and sensor airglow
    window = min(12, max(3, len(df) // 4))
    df['sol_trend_smooth'] = df['sum_of_lights'].rolling(window=window, min_periods=1, center=True).mean()
    df['lit_area_smooth'] = df['lit_area_pct'].rolling(window=window, min_periods=1, center=True).mean()

    # Growth rate
    df['sol_yoy_growth_pct'] = df['sol_trend_smooth'].pct_change(periods=12) * 100.0
    df['sol_yoy_growth_pct'] = df['sol_yoy_growth_pct'].fillna(0.0)

    return df

def detect_expansion_corridors(images, years):
    """
    Computes spatial difference heatmaps showing exact locations of new development
    between baseline (early years) and recent years.
    """
    baseline = np.mean(images[:12], axis=0).astype(np.float32)
    recent = np.mean(images[-12:], axis=0).astype(np.float32)

    diff = recent - baseline
    growth_corridor_mask = np.clip(diff, 0, 255).astype(np.uint8)
    smoothed_growth = cv2.GaussianBlur(growth_corridor_mask, (15, 15), 3.0)

    return {
        'baseline_year': years[0],
        'recent_year': years[-1],
        'radiance_difference': diff,
        'smoothed_growth_corridor': smoothed_growth
    }
