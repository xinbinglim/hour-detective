# hour-detective

My entry for Hack Club Haggle - Hour Detective Challenge!
Predict how many hours a project really took, based on it's git history, code, README, live site and more.

## How To Run

run: predict.py

It will writes 'predictions.csv' with columns 'project_id, hours_spent'.
It takes about 25 seconds on my PC (CPU only).

## Others

**Extra Features**
- 'biggest_commit_share': share of all code added in the single biggest commit
- 'lines_per_active_day': code per day
- 'commit_per_active_day'
- 'active_day_ratio': active days / project lifetime

**Models**
- LightGBM
- XGBoost
- CatBoost

Final prediction is the average of three.

## Results (5-fold cross-validation on train.csv, RMSLE, lower is better)

| Version | Score |
|---------|-------|
| Same guess for every project | 0.9623           |
| LightGBM, no extra features  | 0.6623           |
| LightGBM + my 4 features     | 0.6643 (untuned) |
| LightGBM tuned with Optuna   | 0.6554           |
| **Average of LightGBM + XGBoost + CatBoost** | **0.6532** |

