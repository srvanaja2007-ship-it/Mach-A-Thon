"""
Machine Learning Model Architectures for GPS/GNSS Positioning Error Correction.
Supports Random Forest Regressor and XGBoost Regressor.
Predicts local metric North & East errors and outputs corrected WGS84 coordinates.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    XGB_PARAMS,
    RF_PARAMS,
    MODELS_DIR,
    MAX_PHYSICAL_CORRECTION_METERS,
)
from src.feature_engineering import apply_corrections

logger = logging.getLogger(__name__)


class GpsErrorCorrectionModel:
    """
    Dual-target regressor predicting metric positioning errors:
    - Delta North (target_err_N) in meters
    - Delta East (target_err_E) in meters
    
    Transforms predictions back to WGS84 Latitude & Longitude corrections.
    """
    def __init__(self, model_type: str = "xgboost", custom_params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.feature_columns = FEATURE_COLUMNS
        self.target_columns = TARGET_COLUMNS
        
        params = custom_params or (XGB_PARAMS if self.model_type == "xgboost" else RF_PARAMS)
        self.params = params
        
        if self.model_type == "xgboost":
            self.model_N = XGBRegressor(**params)
            self.model_E = XGBRegressor(**params)
        elif self.model_type in ["random_forest", "rf"]:
            self.model_N = RandomForestRegressor(**params)
            self.model_E = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'xgboost' or 'random_forest'.")
            
        self.is_trained = False

    def fit(self, X: pd.DataFrame, y_N: np.ndarray, y_E: np.ndarray) -> "GpsErrorCorrectionModel":
        """
        Fits North and East error regressors.
        """
        X_mat = X[self.feature_columns].values
        logger.info(f"Fitting {self.model_type.upper()} on {len(X_mat):,} samples with {len(self.feature_columns)} features...")
        
        self.model_N.fit(X_mat, y_N)
        self.model_E.fit(X_mat, y_E)
        self.is_trained = True
        logger.info(f"{self.model_type.upper()} model training completed.")
        return self

    def predict_errors(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts metric errors in meters: (pred_err_N, pred_err_E).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        X_mat = X[self.feature_columns].values
        pred_N = self.model_N.predict(X_mat)
        pred_E = self.model_E.predict(X_mat)
        return pred_N, pred_E

    def correct_coordinates(
        self,
        lat_obs: np.ndarray,
        lon_obs: np.ndarray,
        X: pd.DataFrame,
        max_correction_meters: Optional[float] = MAX_PHYSICAL_CORRECTION_METERS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts errors and returns corrected coordinates:
        returns: (lat_corrected, lon_corrected, pred_err_N, pred_err_E)
        """
        pred_N, pred_E = self.predict_errors(X)
        if max_correction_meters is not None and max_correction_meters > 0:
            pred_N = np.clip(pred_N, -max_correction_meters, max_correction_meters)
            pred_E = np.clip(pred_E, -max_correction_meters, max_correction_meters)
            
        lat_corr, lon_corr = apply_corrections(lat_obs, lon_obs, pred_N, pred_E)
        return lat_corr, lon_corr, pred_N, pred_E

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns feature importances for both North and East regressors.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first.")
            
        imp_N = self.model_N.feature_importances_
        imp_E = self.model_E.feature_importances_
        
        df = pd.DataFrame({
            "Feature": self.feature_columns,
            "Importance_North": imp_N,
            "Importance_East": imp_E,
            "Mean_Importance": (imp_N + imp_E) / 2.0,
        }).sort_values("Mean_Importance", ascending=False).reset_index(drop=True)
        return df

    def save(self, save_dir: Optional[Path] = None, prefix: str = "") -> Path:
        """
        Saves trained models, feature list, and metadata to disk.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
            
        out_dir = Path(save_dir or MODELS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        p = f"{prefix}_" if prefix else f"{self.model_type}_"
        model_N_path = out_dir / f"{p}model_N.joblib"
        model_E_path = out_dir / f"{p}model_E.joblib"
        meta_path = out_dir / f"{p}metadata.json"
        
        joblib.dump(self.model_N, model_N_path)
        joblib.dump(self.model_E, model_E_path)
        
        meta = {
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "params": {k: (str(v) if isinstance(v, (type, object)) and not isinstance(v, (int, float, str, bool, list, dict)) else v) for k, v in self.params.items()},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        logger.info(f"Model saved to {out_dir} with prefix '{p}'.")
        return out_dir

    @classmethod
    def load(cls, load_dir: Optional[Path] = None, prefix: str = "xgboost") -> "GpsErrorCorrectionModel":
        """
        Loads trained models and metadata from disk.
        """
        in_dir = Path(load_dir or MODELS_DIR)
        p = f"{prefix}_" if not prefix.endswith("_") else prefix
        meta_path = in_dir / f"{p}metadata.json"
        
        if not meta_path.exists():
            # Try exact prefix
            meta_path = in_dir / f"{prefix}metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Model metadata not found at {meta_path}.")
            p = prefix
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        instance = cls(model_type=meta["model_type"])
        instance.feature_columns = meta["feature_columns"]
        instance.target_columns = meta["target_columns"]
        
        model_N_path = in_dir / f"{p}model_N.joblib"
        model_E_path = in_dir / f"{p}model_E.joblib"
        
        instance.model_N = joblib.load(model_N_path)
        instance.model_E = joblib.load(model_E_path)
        instance.is_trained = True
        logger.info(f"Loaded {meta['model_type']} model successfully from {in_dir}.")
        return instance
