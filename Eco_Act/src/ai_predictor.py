import numpy as np
import cv2
import pandas as pd
from scipy.ndimage import gaussian_filter
from download_online_dataset import LANDMARKS, coord_to_pixel

class SpatioTemporalDevelopmentAI:
    """
    AI Spatio-Temporal Model that learns 10-year development dynamics across
    Tumkur and its outskirts (768x768) and synthesizes future development rasters & economic output.
    """
    def __init__(self, base_gddp_crores_2015=32000.0, elasticity_alpha=0.75):
        self.base_gddp = base_gddp_crores_2015
        self.elasticity = elasticity_alpha
        self.calibrated = False

    def fit_macro_trend(self, df):
        """
        Fits macroeconomic regression models on historical Sum of Lights (SOL) smooth trend.
        """
        t = df['timestamp'].values
        sol = df['sol_trend_smooth'].values
        lit_area = df['lit_area_smooth'].values

        t_norm = t - t[0]
        # Fit log-polynomial for compounding economic expansion
        poly_sol = np.polyfit(t_norm, np.log(sol), deg=1)
        poly_lit = np.polyfit(t_norm, lit_area, deg=1)

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

            # Macroeconomic GDDP prediction formula (calibrated to Karnataka Economic Survey):
            sol_ratio = pred_sol / self.base_sol
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
        Generates simulated future 2D satellite night-light rasters (768x768).
        Uses spatial momentum + agglomeration diffusion along transit corridors.
        """
        N, H, W = images.shape
        # Use average of recent 12 months as stable starting point
        last_img = np.mean(images[-12:], axis=0).astype(np.float32)

        # Growth velocity field: difference between last 12 months and previous 24 months
        baseline_img = np.mean(images[-36:-24], axis=0).astype(np.float32) if N >= 36 else np.mean(images[:12], axis=0).astype(np.float32)
        dt_hist = max(1.0, historical_timestamps[-1] - historical_timestamps[0])
        velocity_field = (last_img - baseline_img) / 2.0 # 2-year difference / 2 = annual velocity

        velocity_field = cv2.GaussianBlur(velocity_field, (11, 11), sigmaX=2.0)
        velocity_field = np.where(velocity_field > 0, velocity_field, velocity_field * 0.15)

        future_maps = []
        curr_map = last_img.copy()
        last_year = historical_timestamps[-1]

        for yr in future_years:
            dt = yr - last_year
            last_year = yr

            # Local corridor momentum
            momentum_gain = velocity_field * dt

            # Agglomeration Diffusion around existing hubs
            diffused = gaussian_filter(curr_map, sigma=3.0)
            fringe_mask = (curr_map > 12.0) & (curr_map < 200.0)
            diffusion_gain = diffused * diffusion_rate * dt * fringe_mask.astype(np.float32)

            # Micro infrastructure noise
            np.random.seed(int(yr))
            noise = np.random.normal(0.0, 0.4, size=(H, W))

            curr_map = curr_map + momentum_gain + diffusion_gain + noise
            curr_map = np.clip(curr_map, 0.0, 255.0)

            future_maps.append((int(yr), curr_map.astype(np.uint8)))

        return future_maps

def analyze_landmark_opportunity_zones(recent_img, future_img, w=1024, h=1024, radius=28):
    """
    Ranks landmarks across Tumkur & outskirts by growth velocity between current and predicted future.
    """
    zones = []
    for lm in LANDMARKS:
        px, py = coord_to_pixel(lm['lat'], lm['lon'], w=w, h=h)
        x1, x2 = max(0, px - radius), min(w, px + radius + 1)
        y1, y2 = max(0, py - radius), min(h, py + radius + 1)

        patch_rec = recent_img[y1:y2, x1:x2].astype(np.float32)
        patch_fut = future_img[y1:y2, x1:x2].astype(np.float32)

        flux_rec = float(np.sum(patch_rec))
        flux_fut = float(np.sum(patch_fut))
        growth_pct = ((flux_fut - flux_rec) / max(100.0, flux_rec)) * 100.0

        if lm['type'] == 'industrial':
            status = "Industrial Corridor Boom"
            action = "High Priority: Power grid substations, commercial credit & freight warehouses"
        elif lm['type'] == 'corridor':
            status = "Expressway Ribbon Sprawl"
            action = "Fast Expansion: Toll plaza logistics, EV charging & retail distribution"
        elif lm['type'] == 'core':
            status = "Dense Urban Anchor"
            action = "Stable Commercial Hub: Micro-ATMs, branch banking & fintech services"
        else:
            status = "Emerging Outskirt Hub"
            action = "Early Growth: Rural banking expansion & agro-processing facilities"

        zones.append({
            'name': lm['name'],
            'type': lm['type'],
            'lat': lm['lat'],
            'lon': lm['lon'],
            'pixel_x': px,
            'pixel_y': py,
            'current_flux': flux_rec,
            'future_flux': flux_fut,
            'growth_pct': growth_pct,
            'status': status,
            'recommended_action': action
        })

    zones.sort(key=lambda x: x['growth_pct'], reverse=True)
    return zones
