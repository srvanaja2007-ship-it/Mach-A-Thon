# GPS/GNSS Model Accuracy Evaluation Report

## Dataset & Leakage Audit
- **Test GPS Observations**: 17,385
- **Test Trajectories**: 7
- **Distance Metric**: WGS84 Ellipsoidal Geodesic Distance (Vincenty inverse formula, metres)
- **Data Leakage Check**: PASSED (0 overlapping train/test trips, 0 ground-truth features in model inputs)

## Model Accuracy Comparison

| Model | MAE (m) | RMSE (m) | Median Error (m) | 95th Percentile Error (m) | RMSE Improvement (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Raw GPS (Observed) | 2.1315 | 2.5493 | 1.7827 | 5.1698 | +0.00% |
| Random Forest | 1.7336 | 2.0666 | 1.6022 | 3.7187 | +18.93% |
| XGBoost | 1.7793 | 2.1190 | 1.6486 | 3.8394 | +16.88% |

## Summary
- **Raw GPS RMSE**: `2.5493 m`
- **Random Forest Corrected RMSE**: `2.0666 m` (`+18.93%`)
- **XGBoost Corrected RMSE**: `2.1190 m` (`+16.88%`)
- **Better Performing Model**: **Random Forest**

## Per-Trajectory RMSE Breakdown

| Trajectory (`tripId`) | Obs | Raw RMSE (m) | RF RMSE (m) | XGB RMSE (m) | RF Imp (%) | XGB Imp (%) | Winner |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `2020-07-24-US-MTV-1/GooglePixel4XL` | 2050 | 1.4586 | 1.8386 | 1.9670 | -26.05% | -34.85% | Random Forest |
| `2020-07-24-US-MTV-1/GooglePixel5` | 2059 | 1.3264 | 1.4005 | 1.4417 | -5.58% | -8.69% | Random Forest |
| `2021-03-16-US-MTV-2/XiaomiMi8` | 2160 | 5.0438 | 3.5602 | 3.6566 | +29.41% | +27.50% | Random Forest |
| `2021-04-02-US-SJC-1/GooglePixel5` | 2283 | 2.4635 | 1.6502 | 1.7687 | +33.01% | +28.20% | Random Forest |
| `2021-07-01-US-MTV-1/GooglePixel5` | 2556 | 1.6267 | 1.9132 | 1.9087 | -17.61% | -17.33% | XGBoost |
| `2021-08-24-US-SVL-1/GooglePixel5` | 3137 | 2.1419 | 1.7265 | 1.7282 | +19.40% | +19.32% | Random Forest |
| `2021-08-24-US-SVL-1/SamsungGalaxyS20Ultra` | 3140 | 2.1887 | 1.8701 | 1.8842 | +14.56% | +13.91% | Random Forest |
