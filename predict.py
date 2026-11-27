import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

START = time.perf_counter()
HERE = Path(__file__).resolve().parent
LABELS = ["hours_spent", "log_hours", "hours_bucket"]
TIME_COLS = ["stars", "forks", "num_issues", "num_pull_requests",
             "num_releases", "author_account_age_years"]
SHIFT = -0.1
SCALE = 0.8

def add_features(df):
    df = df.copy()
    df["biggest_commit_share"] = df["max_lines_in_one_commit"] / df["lines_added_total"].replace(0, np.nan)
    df["lines_per_active_day"] = df["lines_of_code"] / df["active_commit_days"].replace(0, np.nan)
    df["commits_per_active_day"] = df["num_commits"] / df["active_commit_days"].replace(0, np.nan)
    df["active_day_ratio"] = df["active_commit_days"] / (df["project_span_days"] + 1)
    return df

ORDER_MAPS = {
    "author_prior_projects":     {"0": 0, "1-2": 1, "3-5": 2, "6+": 3},
    "author_account_age_years":  {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3},
}
def encode_ordered(df):
    df = df.copy()
    for col, mapping in ORDER_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping)
    return df

def encode_yes_no(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        values = set(df[col].dropna().astype(str).unique())
        if values <= {"True", "False"}:
            df[col] = df[col].astype(str).map({"True": 1.0, "False": 0.0})
    return df

def encode_categories(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df

def prepare(df):
    df = add_features(df)
    df = encode_ordered(df)
    df = encode_yes_no(df)
    df = encode_categories(df)
    return df

BEST_LGB = {
    "n_estimators": 865, "learning_rate": 0.012964449902814406, "num_leaves": 35,
        "min_child_samples": 11, "subsample": 0.870297694890708,
        "colsample_bytree": 0.5828255385942684, "reg_lambda": 0.002402089003977252,
}

BEST_XGB = {
    "n_estimators": 814, "learning_rate": 0.010311069581219252, "max_depth": 9,
    "min_child_weight": 7.614778905496447, "subsample": 0.5864764800801583,
    "colsample_bytree": 0.3503811143222532, "reg_lambda": 0.024691999110864553,
}

BEST_CAT = {
    "iterations": 849, "learning_rate": 0.037576200996845, "depth": 8,
    "l2_leaf_reg": 2.7913342621237325, "random_strength": 1.100700356059648,
}

def main():
    train = pd.read_csv(HERE / "train.csv")
    test = pd.read_csv(HERE / "test.csv")

    y = np.log1p(train["hours_spent"])
    X_train = train.drop(columns=LABELS + ["project_id"] + TIME_COLS)
    X_test = test.drop(columns=LABELS + ["project_id"] + TIME_COLS, errors="ignore")

    both = pd.concat([X_train, X_test], keys=["train", "test"])
    both = prepare(both)
    bool_cols = both.select_dtypes(include="bool").columns
    both = both.astype({c: int for c in bool_cols})

    cat_cols = both.select_dtypes(include="category").columns.tolist()
    both_cat = both.copy()
    for c in cat_cols:
        both_cat[c] = both_cat[c].astype(object).fillna("missing").astype(str)

    X_tr, X_te = both.loc["train"], both.loc["test"]
    Xc_tr, Xc_te = both_cat.loc["train"], both_cat.loc["test"]

    SEEDS = [42, 1, 2]
    lgb_preds, xgb_preds, cat_preds = [], [], []
    for seed in SEEDS:
        lgb_model = lgb.LGBMRegressor(**BEST_LGB, subsample_freq=1, random_state=seed, verbose=-1)
        xgb_model = xgb.XGBRegressor(**BEST_XGB, enable_categorical=True,
                                     tree_method="hist", random_state=seed)
        cat_model = CatBoostRegressor(**BEST_CAT, cat_features=cat_cols, random_seed=seed,
                                      thread_count=-1, verbose=0, allow_writing_files=False)

        lgb_model.fit(X_tr, y)
        xgb_model.fit(X_tr, y)
        cat_model.fit(Xc_tr, y)

        lgb_preds.append(lgb_model.predict(X_te))
        xgb_preds.append(xgb_model.predict(X_te))
        cat_preds.append(cat_model.predict(Xc_te))
        print(f"Seed {seed} done ({time.perf_counter() - START:.1f} s so far)")

    pred_log = (0.5 * np.mean(xgb_preds, axis=0)
              + 0.3 * np.mean(lgb_preds, axis=0)
              + 0.2 * np.mean(cat_preds, axis=0))
    pred_log = pred_log + SHIFT
    center = pred_log.mean()
    pred_log = center + SCALE * (pred_log - center)
    pred_hours = np.expm1(pred_log).clip(0, 140.7)
    
    out = pd.DataFrame({"project_id": test["project_id"], "hours_spent": pred_hours})
    out.to_csv(HERE / "predictions.csv", index=False)
    print(f"Wrote {len(out)} predictions in {time.perf_counter() - START:.1f} seconds")

if __name__ == "__main__":
    main()