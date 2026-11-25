import numpy as np
import cv2
import pandas as pd
from scipy.ndimage import gaussian_filter


class SpatioTemporalDevelopmentAI:
    """
    AI Spatio-Temporal Model that learns multi-year development dynamics across
    any satellite night-light dataset and synthesizes future development rasters & economic output.
    """
    def __init__(self, base_gddp_crores=32000.0, elasticity_alpha=0.75, base_gddp_crores_2015=None):
        self.base_gddp = base_gddp_crores_2015 if base_gddp_crores_2015 is not None else base_gddp_crores
        self.elasticity = elasticity_alpha
        self.calibrated = False

    def fit_macro_trend(self, df):
        """
        Fits macroeconomic regression models on historical Sum of Lights (SOL) smooth trend.
        """
        t = df['timestamp'].values.astype(np.float64)
        sol = np.maximum(1.0, df['sol_trend_smooth'].values.astype(np.float64))
        lit_area = df['lit_area_smooth'].values.astype(np.float64)

        t_norm = t - t[0]
        if len(t_norm) >= 2 and np.ptp(t_norm) > 0:
            poly_sol = np.polyfit(t_norm, np.log(sol), deg=1)
            poly_lit = np.polyfit(t_norm, lit_area, deg=1)
        else:
            poly_sol = np.array([0.0, np.log(sol[0])])
            poly_lit = np.array([0.0, lit_area[0]])

        self.t0 = t[0]
        self.poly_sol = poly_sol
        self.poly_lit = poly_lit
        self.base_sol = sol[0]
        self.calibrated = True
        return self

    def predict_macro_future(self, future_years):
        """
        Predicts annual economic output (GDDP in ₹ Crores) and lit area for future years.
        """
        if not self.calibrated:
            raise ValueError("Model must be fitted first.")

        records = []
        for yr in future_years:
            t = yr - self.t0
            pred_log_sol = np.polyval(self.poly_sol, t)
            pred_sol = float(np.exp(pred_log_sol))
            pred_lit_pct = float(max(1.0, min(100.0, np.polyval(self.poly_lit, t))))

            sol_ratio = pred_sol / max(1.0, self.base_sol)
            pred_gddp = float(self.base_gddp * (sol_ratio ** self.elasticity))
            econ_index = float(min(100.0, 35.0 * (pred_gddp / self.base_gddp)))

            records.append({
                'year': int(yr),
                'predicted_sum_of_lights': pred_sol,
                'predicted_lit_area_pct': pred_lit_pct,
                'predicted_gddp_crores': pred_gddp,
                'economic_activity_index': econ_index,
                'confidence_interval_low_gddp': pred_gddp * 0.93,
                'confidence_interval_high_gddp': pred_gddp * 1.07,
                'is_prediction': True
            })

        return pd.DataFrame(records)

    def predict_future_raster_maps(self, images, historical_timestamps, future_years, diffusion_rate=0.06):
        """
        Generates simulated future 2D satellite night-light rasters at native resolution.
        Uses spatial momentum + agglomeration diffusion along transit corridors.
        """
        N, H, W = images.shape
        recent_k = min(12, max(1, N // 3))
        last_img = np.mean(images[-recent_k:], axis=0).astype(np.float32)

        if N >= 36:
            baseline_img = np.mean(images[-36:-24], axis=0).astype(np.float32)
            dt_span = 2.0
        elif N >= 2:
            half = max(1, N // 2)
            baseline_img = np.mean(images[:half], axis=0).astype(np.float32)
            dt_span = max(1.0, (historical_timestamps[-1] - historical_timestamps[0]) / 2.0)
        else:
            baseline_img = last_img.copy()
            dt_span = 1.0

        velocity_field = (last_img - baseline_img) / dt_span
        velocity_field = cv2.GaussianBlur(velocity_field, (11, 11), sigmaX=2.0)
        velocity_field = np.where(velocity_field > 0, velocity_field, velocity_field * 0.15)

        future_maps = []
        curr_map = last_img.copy()
        last_year = historical_timestamps[-1]

        for yr in future_years:
            dt = yr - last_year
            last_year = yr

            momentum_gain = velocity_field * dt
            diffused = gaussian_filter(curr_map, sigma=3.0)
            fringe_mask = (curr_map > 12.0) & (curr_map < 200.0)
            diffusion_gain = diffused * diffusion_rate * dt * fringe_mask.astype(np.float32)

            np.random.seed(int(yr))
            noise = np.random.normal(0.0, 0.4, size=(H, W))

            curr_map = np.clip(curr_map + momentum_gain + diffusion_gain + noise, 0.0, 255.0)
            future_maps.append((int(yr), curr_map.astype(np.uint8)))

        return future_maps


def _classify_zone(zone_type, growth_pct):
    if zone_type == 'industrial':
        return "Industrial Corridor Boom", "High Priority: Power grid substations, commercial credit & freight warehouses"
    if zone_type == 'corridor':
        return "Expressway Ribbon Sprawl", "Fast Expansion: Toll plaza logistics, EV charging & retail distribution"
    if zone_type == 'core':
        return "Dense Urban Anchor", "Stable Commercial Hub: Micro-ATMs, branch banking & fintech services"
    if growth_pct > 20.0:
        return "Rapid Growth Hotspot", "High Priority: Infrastructure, power substations & commercial expansion"
    if growth_pct > 8.0:
        return "Emerging Growth Zone", "Targeted Investment: Road connectivity, SME parks & utility upgrades"
    return "Stable / Low-Growth Zone", "Monitor & Sustain: Micro-grid energy & rural enterprise support"


def analyze_opportunity_zones(recent_img, future_img, landmark_info=None, w=None, h=None, radius=28):
    """
    Ranks opportunity zones by growth velocity between current and predicted future.
    Uses landmark_info if provided; otherwise auto-discovers top spatial grid zones.
    """
    H, W = recent_img.shape[:2]
    w = w or W
    h = h or H
    zones = []

    if landmark_info:
        landmarks = landmark_info.get('landmarks', []) if isinstance(landmark_info, dict) else landmark_info
        coord_fn = landmark_info.get('coord_to_pixel') if isinstance(landmark_info, dict) else None
        scaled_r = max(4, int(radius * min(W, H) / 1024.0))

        for lm in landmarks:
            if 'pixel_x' in lm and 'pixel_y' in lm:
                px, py = int(lm['pixel_x']), int(lm['pixel_y'])
            elif coord_fn is not None and 'lat' in lm and 'lon' in lm:
                px, py = coord_fn(lm['lat'], lm['lon'], w=W, h=H)
            elif 'x_norm' in lm and 'y_norm' in lm:
                px, py = int(lm['x_norm'] * W), int(lm['y_norm'] * H)
            else:
                continue

            x1, x2 = max(0, px - scaled_r), min(W, px + scaled_r + 1)
            y1, y2 = max(0, py - scaled_r), min(H, py + scaled_r + 1)

            flux_rec = float(np.sum(recent_img[y1:y2, x1:x2].astype(np.float32)))
            flux_fut = float(np.sum(future_img[y1:y2, x1:x2].astype(np.float32)))
            growth_pct = ((flux_fut - flux_rec) / max(100.0, flux_rec)) * 100.0

            status, action = _classify_zone(lm.get('type', 'town'), growth_pct)
            zones.append({
                'name': lm['name'],
                'type': lm.get('type', 'zone'),
                'lat': lm.get('lat', 0.0),
                'lon': lm.get('lon', 0.0),
                'pixel_x': px,
                'pixel_y': py,
                'current_flux': flux_rec,
                'future_flux': flux_fut,
                'growth_pct': growth_pct,
                'status': status,
                'recommended_action': action
            })
    else:
        # Auto-discover 4x4 spatial grid zones directly from the imagery
        rows, cols = 4, 4
        row_names = ["North", "North-Central", "South-Central", "South"]
        col_names = ["West", "West-Central", "East-Central", "East"]
        cell_h, cell_w = max(1, H // rows), max(1, W // cols)

        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h if r < rows - 1 else H
                x1, x2 = c * cell_w, (c + 1) * cell_w if c < cols - 1 else W
                patch_rec = recent_img[y1:y2, x1:x2].astype(np.float32)
                patch_fut = future_img[y1:y2, x1:x2].astype(np.float32)

                flux_rec = float(np.sum(patch_rec))
                flux_fut = float(np.sum(patch_fut))
                growth_pct = ((flux_fut - flux_rec) / max(100.0, flux_rec)) * 100.0
                mean_rec = float(np.mean(patch_rec))

                if mean_rec > 120:
                    z_type = 'core'
                elif growth_pct > 15:
                    z_type = 'industrial'
                else:
                    z_type = 'outskirt'

                status, action = _classify_zone(z_type, growth_pct)
                sector_label = "Central Core" if (r in (1, 2) and c in (1, 2)) else f"{row_names[r]} {col_names[c]}"
                zones.append({
                    'name': f"Sector {r+1}-{c+1} ({sector_label})",
                    'type': z_type,
                    'lat': 0.0,
                    'lon': 0.0,
                    'pixel_x': (x1 + x2) // 2,
                    'pixel_y': (y1 + y2) // 2,
                    'current_flux': flux_rec,
                    'future_flux': flux_fut,
                    'growth_pct': growth_pct,
                    'status': status,
                    'recommended_action': action
                })

    zones.sort(key=lambda x: x['growth_pct'], reverse=True)
    return zones


def analyze_landmark_opportunity_zones(recent_img, future_img, w=1024, h=1024, radius=28, landmark_info=None):
    """Backward-compatible wrapper."""
    return analyze_opportunity_zones(recent_img, future_img, landmark_info=landmark_info, w=w, h=h, radius=radius)
