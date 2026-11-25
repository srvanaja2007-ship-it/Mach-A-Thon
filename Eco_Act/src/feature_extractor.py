import numpy as np
import cv2
import pandas as pd


def _get_landmark_pixels(lm, w, h, coord_fn=None):
    if 'pixel_x' in lm and 'pixel_y' in lm:
        return int(lm['pixel_x']), int(lm['pixel_y'])
    if coord_fn is not None and 'lat' in lm and 'lon' in lm:
        return coord_fn(lm['lat'], lm['lon'], w=w, h=h)
    # Relative normalized coords [0..1]
    if 'x_norm' in lm and 'y_norm' in lm:
        return int(lm['x_norm'] * w), int(lm['y_norm'] * h)
    return w // 2, h // 2


def extract_landmark_radiance(img, w=1024, h=1024, radius=24, landmark_info=None):
    """
    Extracts integrated radiant flux around each landmark if landmark_info is given,
    and always extracts generic 5-zone spatial quadrant flux (Core + NW/NE/SW/SE).
    """
    results = {}

    # Generic spatial sectors (works for ANY dataset without coordinates)
    mid_y, mid_x = h // 2, w // 2
    q_h, q_w = max(1, h // 4), max(1, w // 4)
    sectors = {
        'central_core': img[mid_y - q_h:mid_y + q_h, mid_x - q_w:mid_x + q_w],
        'north_west':   img[:mid_y, :mid_x],
        'north_east':   img[:mid_y, mid_x:],
        'south_west':   img[mid_y:, :mid_x],
        'south_east':   img[mid_y:, mid_x:],
    }
    for s_name, patch in sectors.items():
        results[f"flux_{s_name}"] = float(np.sum(patch))
        results[f"mean_{s_name}"] = float(np.mean(patch))

    if landmark_info:
        landmarks = landmark_info.get('landmarks', []) if isinstance(landmark_info, dict) else landmark_info
        coord_fn = landmark_info.get('coord_to_pixel') if isinstance(landmark_info, dict) else None
        scaled_r = max(4, int(radius * min(w, h) / 1024.0))
        for lm in landmarks:
            px, py = _get_landmark_pixels(lm, w, h, coord_fn)
            x1, x2 = max(0, px - scaled_r), min(w, px + scaled_r + 1)
            y1, y2 = max(0, py - scaled_r), min(h, py + scaled_r + 1)
            patch = img[y1:y2, x1:x2]
            key = lm['name'].lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            results[f"flux_{key}"] = float(np.sum(patch))
            results[f"mean_{key}"] = float(np.mean(patch)) if patch.size else 0.0

    return results


def extract_geospatial_features(images, timestamps, years, months, lit_threshold=15.0, high_ind_threshold=70.0, landmark_info=None):
    """
    Extracts multi-scale geospatial radiance and development metrics for all time steps.
    """
    N, H, W = images.shape
    records = []
    yy, xx = np.mgrid[0:H, 0:W]

    print(f"[FeatureExtractor] Extracting spatial & radiant flux dynamics across {N} frames ({W}x{H})...")

    for i in range(N):
        img = images[i].astype(np.float32)
        t_dec = timestamps[i]
        yr = years[i]
        mo = months[i]

        sol = float(np.sum(img))
        mean_rad = float(np.mean(img))
        max_rad = float(np.max(img))

        lit_mask = (img > lit_threshold).astype(np.uint8)
        lit_pixels = int(np.sum(lit_mask))
        lit_area_pct = float((lit_pixels / (H * W)) * 100.0)

        high_ind_mask = (img > high_ind_threshold).astype(np.uint8)
        high_ind_pixels = int(np.sum(high_ind_mask))

        total_w = np.sum(img) + 1e-6
        com_x = float(np.sum(xx * img) / total_w)
        com_y = float(np.sum(yy * img) / total_w)

        lm_flux = extract_landmark_radiance(img, w=W, h=H, landmark_info=landmark_info)

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

    window = min(12, max(1, len(df) // 3))
    df['sol_trend_smooth'] = df['sum_of_lights'].rolling(window=window, min_periods=1, center=True).mean()
    df['lit_area_smooth'] = df['lit_area_pct'].rolling(window=window, min_periods=1, center=True).mean()

    growth_period = min(12, max(1, len(df) - 1)) if len(df) > 1 else 1
    df['sol_yoy_growth_pct'] = (df['sol_trend_smooth'].pct_change(periods=growth_period) * 100.0).fillna(0.0)

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
