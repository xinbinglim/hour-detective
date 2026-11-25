# hour-detective

Predict how many hours a project really took, based on it's git history, code, README, live site and more.

## How To Run

run: predict.py

It will writes 'predictions.csv' with columns 'project_id, hours_spent'.
It takes about 105 seconds on my PC (CPU only).

## Others

**Extra Features**
- biggest_commit_share
- lines_per_active_day
- active_day_ratio

**Models**
- LightGBM and XGBoost (tuned with Optuna)
- CatBoost

Each is trained with 3 random seeds and averaged.
Final prediction = 0.5 × XGBoost + 0.3 × LightGBM + 0.2 × CatBoost.

Target encoding: CV 0.6546 → 0.6568, slightly worse. 
Stretching the final predictions with a linear fit: worse.

## Results (RMSLE, lower is better)

| Version | CV score |
|---|---|
| Baseline | 0.9623 |
| LightGBM (tuned, with 4 features) | 0.6554 |
| XGBoost (tuned (3 seeds)) | 0.6525 |
| Equal blend of LGB + XGB + Cat | 0.6532 |
| **Final: weighted blend, 3 seeds** | **0.6524** |

| Ship | Change | Live RMSLE |
|---|---|---|
| #1 | Blend of 3 models | 0.7549 |
| #2 | 3-seed averaging | 0.7510 |
| #3 | Tuned XGBoost + weighted blend | 0.7487 |