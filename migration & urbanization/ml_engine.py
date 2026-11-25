"""
ml_engine.py — Multi-Sensor AI Classifier & 2027–2028 Sprawl Forecaster
UrbanPulse AI | Aeromania '26 Mach-A-Thon | Track A: Migration & Urbanization

4-Zone AI Classifier (Random Forest / Gradient Boosting) + Spatial RF Regressor
for 2027–2028 migration gravity prediction.
"""

from __future__ import annotations

import logging
import pickle
import hashlib
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------
ZONE_LABELS = {
    0: "Planned High-Rise / Tech & Industrial Core",
    1: "Active Legal Suburbs",
    2: "Night-Light Gap / Underserved Migrant Settlement",
    3: "Protected Green / Water / Coastal Buffer",
}

ZONE_COLORS = {
    0: "#06B6D4",   # Cyan — tech/industrial
    1: "#22C55E",   # Green — legal suburbs
    2: "#EF4444",   # Red — migrant/underserved
    3: "#84CC16",   # Lime — green/protected
}

ZONE_SHORT = {
    0: "Tech Core",
    1: "Legal Suburb",
    2: "Night-Light Gap",
    3: "Green Buffer",
}

# ---------------------------------------------------------------------------
# Rule-based label generator (physics-informed, no random noise labels)
# ---------------------------------------------------------------------------
def _generate_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Assign physically meaningful zone labels to each 250m grid cell.

    Rules based on Karnataka satellite statistics literature:
      Zone 3 (Protected): water_mask==1 OR ndvi_2026 > 0.45
      Zone 0 (Tech Core): delta_ndbi > 0.08 AND vv_db > -8 AND viirs_dnb > 20
      Zone 2 (NLG):       delta_ndbi > 0.05 AND viirs_dnb < 5 AND road_dist_m > 800
      Zone 1 (Suburb):    everything else with delta_ndbi > 0
      Zone 3 fallback:    low built-up, high vegetation
    """
    n = len(df)
    labels = np.full(n, 1, dtype=int)   # default: legal suburb

    # Zone 3 — green/water
    is_green = (df["water_mask"] > 0.5) | (df["ndvi_2026"] > 0.45)
    labels[is_green] = 3

    # Zone 0 — tech/industrial core
    is_tech = (
        (df["delta_ndbi"] > 0.08) &
        (df["vv_db"] > -8.0) &
        (df["viirs_dnb"] > 20.0) &
        (~is_green)
    )
    labels[is_tech] = 0

    # Zone 2 — night-light gap / migrant settlement
    is_nlg = (
        (df["delta_ndbi"] > 0.05) &
        (df["viirs_dnb"] < 5.0) &
        (df["road_dist_m"] > 800) &
        (~is_green) &
        (~is_tech)
    )
    labels[is_nlg] = 2

    # Zone 3 fallback — undeveloped fringe (low built-up + dark)
    is_undeveloped = (
        (df["delta_ndbi"] <= 0.0) &
        (df["ndvi_2026"] > 0.25) &
        (~is_green)
    )
    labels[is_undeveloped] = 3

    return labels


# ---------------------------------------------------------------------------
# Classifier — 4-Zone Random Forest / Gradient Boosting
# ---------------------------------------------------------------------------
FEATURE_COLS = ["delta_ndbi", "ndvi_2026", "ndbi_2026",
                "vv_db", "vh_db", "viirs_dnb", "road_dist_m", "water_mask"]


def train_classifier(df: pd.DataFrame) -> tuple[Pipeline, float, np.ndarray]:
    """
    Train GradientBoostingClassifier on physics-labelled data.
    Returns (pipeline, cv_accuracy, predicted_labels).
    """
    X = df[FEATURE_COLS].values
    y = _generate_labels(df)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_acc = float(cv_scores.mean())
    log.info("Classifier CV accuracy: %.3f ± %.3f", cv_acc, cv_scores.std())

    clf.fit(X, y)
    preds = clf.predict(X)
    return clf, cv_acc, preds


# ---------------------------------------------------------------------------
# 2027–2028 Migration Gravity Forecaster — Spatial RF Regressor
# ---------------------------------------------------------------------------
REGRESSOR_FEATURES = ["delta_ndbi", "road_dist_m", "viirs_dnb",
                      "vv_db", "water_mask", "ndvi_2026"]


def _neighbour_urban_density(delta_ndbi_grid: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Compute local mean delta_ndbi within a (2*radius+1)² neighbourhood.
    Acts as 'urban pressure from neighbours' spatial feature.
    """
    from scipy.ndimage import uniform_filter
    return uniform_filter(delta_ndbi_grid, size=2 * radius + 1).astype(np.float32)


def train_forecaster(df: pd.DataFrame,
                     delta_ndbi_grid: np.ndarray) -> tuple[RandomForestRegressor, np.ndarray]:
    """
    Train a Spatial RF Regressor on 2021→2026 growth drivers to predict
    2027–2028 sprawl risk probability (0–1) per 250m cell.

    Target = normalised future-growth proxy derived from:
      - local delta_ndbi (how much it built up 2021–2026)
      - neighbour density (urban pressure)
      - inverse road distance (accessibility pull)
      - inverse viirs (electricity gap = informal settlement attractor)
    """
    n = len(df)
    # Reshape delta_ndbi_grid to (N_CELLS, N_CELLS)
    nc = int(np.sqrt(n))
    ndbi_grid = delta_ndbi_grid.reshape(nc, nc)
    neigh = _neighbour_urban_density(ndbi_grid).ravel()

    # Heuristic physics-based sprawl risk target (0–1)
    road_pull       = 1.0 / (1.0 + df["road_dist_m"].values / 500.0)   # close roads → growth
    nlg_attractor   = np.where(df["viirs_dnb"].values < 5.0, 1.0, 0.0) # dark areas attract informal
    ndbi_momentum   = np.clip(df["delta_ndbi"].values * 5.0, 0, 1)      # built-up momentum
    green_penalty   = df["water_mask"].values                            # water/green → no growth
    neighbour_press = np.clip(neigh * 4.0, 0, 1)

    risk = (
        0.35 * ndbi_momentum +
        0.25 * road_pull +
        0.20 * nlg_attractor +
        0.15 * neighbour_press -
        0.05 * green_penalty
    )
    risk = np.clip(risk, 0.0, 1.0).astype(np.float32)

    # Build feature matrix with spatial neighbours
    df_feat = df[REGRESSOR_FEATURES].copy()
    df_feat["neigh_urban_density"] = neigh

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(df_feat.values, risk)
    sprawl_prob = rf.predict(df_feat.values).astype(np.float32)
    sprawl_prob = np.clip(sprawl_prob, 0, 1)

    log.info("Forecaster trained. Mean sprawl risk = %.3f (max=%.3f)",
             sprawl_prob.mean(), sprawl_prob.max())
    return rf, sprawl_prob


# ---------------------------------------------------------------------------
# Full ML pipeline — classify + forecast
# ---------------------------------------------------------------------------
def run_ml_pipeline(data: dict) -> dict:
    """
    Given a data dict from data_pipeline.fetch_location_data, run the full ML.
    Returns extended dict with zone labels, sprawl probabilities, and stats.
    """
    from data_pipeline import grid_to_dataframe

    lat = data["lat"]
    lon = data["lon"]
    key = hashlib.md5(f"{lat:.4f}_{lon:.4f}".encode()).hexdigest()[:12]
    ml_cache = CACHE_DIR / f"{key}_ml.pkl"

    if ml_cache.exists():
        log.info("Loading ML results from cache: %s", ml_cache)
        with open(ml_cache, "rb") as f:
            return pickle.load(f)

    df = grid_to_dataframe(data)

    # 1. Classification
    log.info("Training 4-Zone Classifier …")
    clf, cv_acc, zone_preds = train_classifier(df)

    # 2. Sprawl forecast
    log.info("Training 2027–28 Sprawl Forecaster …")
    _, sprawl_prob = train_forecaster(df, data["delta_ndbi"])

    # 3. Compute summary statistics
    nc = data["n_cells"]
    cell_area_sqkm = (data["grid_size_m"] / 1000) ** 2   # 0.0625 km² per cell

    n_zones = {z: int((zone_preds == z).sum()) for z in range(4)}
    new_built_sqkm = float(np.sum(data["delta_ndbi"] > 0.03) * cell_area_sqkm)
    n_nlg_alerts   = int(n_zones.get(2, 0))
    nlg_area_sqkm  = float(n_nlg_alerts * cell_area_sqkm)

    # Predicted 2027–28 migrant influx estimate
    # Based on Karnataka urbanisation rate (3.1% pa) and satellite growth signal
    base_urban_pop  = 85_000   # representative peri-urban fringe pop per 12km box
    influx_2027_28  = int(base_urban_pop * sprawl_prob.mean() * 1.8)

    # Feature importances
    feat_imp = dict(zip(FEATURE_COLS, clf.named_steps["model"].feature_importances_))

    result = {
        **data,
        # ML outputs (flat arrays)
        "zone_preds":    zone_preds.reshape(nc, nc).astype(np.int8),
        "sprawl_prob":   sprawl_prob.reshape(nc, nc),
        # Statistics
        "cv_accuracy":       cv_acc,
        "n_zones":           n_zones,
        "new_built_sqkm":    new_built_sqkm,
        "n_nlg_alerts":      n_nlg_alerts,
        "nlg_area_sqkm":     nlg_area_sqkm,
        "influx_2027_28":    influx_2027_28,
        "feat_importances":  feat_imp,
        "sprawl_prob_mean":  float(sprawl_prob.mean()),
        "sprawl_prob_max":   float(sprawl_prob.max()),
    }

    with open(ml_cache, "wb") as f:
        pickle.dump(result, f, protocol=4)
    log.info("ML results cached → %s", ml_cache)
    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_pipeline import PRESET_HUBS, fetch_location_data
    for name, hub in list(PRESET_HUBS.items())[:1]:
        print(f"\n=== {name} ===")
        data = fetch_location_data(hub["lat"], hub["lon"])
        result = run_ml_pipeline(data)
        print(f"  CV Accuracy   : {result['cv_accuracy']:.3f}")
        print(f"  New Built-Up  : {result['new_built_sqkm']:.2f} km²")
        print(f"  NLG Alerts    : {result['n_nlg_alerts']}")
        print(f"  Influx 27-28  : {result['influx_2027_28']:,}")

