import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
import os

def main():
    feature_dict = {}
    for f in ['circuit_features_master.csv', 'deep_validation_results.csv']:
        if os.path.exists(f):
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                m = row.get("model_type", row.get("model"))
                s = int(row.get("parameter_size", row.get("size", 0)))
                if "total_assignments" in df.columns and not pd.isna(row["total_assignments"]):
                    feature_dict[(m, s)] = {
                        "domain_size": row.get("domain_size", 0),
                        "total_assignments": row["total_assignments"],
                        "lookup_span": row.get("lookup_span", 0),
                        "total_const_size": row.get("total_const_size", 0)
                    }

    all_files = ['circuit_features_master.csv', 'deep_validation_results.csv',
                 'high_res_rigorous_results.csv', 'transformer_block_results.csv']

    dataset_rows = []
    for f in all_files:
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        time_col = next((c for c in ["actual_proving_time", "proving_time", "time"] if c in df.columns), None)
        if not time_col: continue

        for _, row in df.iterrows():
            m = row.get("model_type", row.get("model"))
            s = int(row.get("parameter_size", row.get("size", 0)))
            t = row[time_col]
            if pd.isna(t): continue

            if (m, s) in feature_dict:
                feat = feature_dict[(m, s)]
                d_size = feat["domain_size"]
                fft_work = d_size * np.log2(d_size) if d_size > 0 else 0.0
                dataset_rows.append({
                    "model_type": str(m),
                    "size": s,
                    "group_id": f"{m}_{s}",  # Unique ID for grouping
                    "actual_proving_time": t,
                    "fft_work": fft_work,
                    "msm_work": float(d_size),
                    "total_assignments": feat["total_assignments"],
                    "lookup_span": feat["lookup_span"],
                    "total_const_size": feat["total_const_size"]
                })

    df_all = pd.DataFrame(dataset_rows).drop_duplicates(subset=["model_type", "size", "actual_proving_time"])
    feature_cols = ["fft_work", "msm_work", "total_assignments", "lookup_span", "total_const_size"]
    X = df_all[feature_cols].values
    y = df_all["actual_proving_time"].values
    groups = df_all["group_id"].values

    print("=" * 80)
    print(" METHOD 1: REPEATED 5-FOLD CV (10 SEEDS = 50 TRAIN/TEST SPLITS) ")
    print("=" * 80)
    rkf = RepeatedKFold(n_splits=5, n_repeats=100000, random_state=42)
    repeated_mapes = []

    for train_idx, val_idx in rkf.split(X):
        w_fold, _ = nnls(X[train_idx], y[train_idx])
        preds = X[val_idx] @ w_fold
        errs = np.abs(y[val_idx] - preds) / y[val_idx] * 100
        repeated_mapes.append(np.mean(errs))

    print(f"--> Repeated K-Fold Mean MAPE: {np.mean(repeated_mapes):.2f}%")
    print(f"--> Standard Deviation:        {np.std(repeated_mapes):.2f}%")
    print(f"--> Min/Max Split MAPE Range:  [{np.min(repeated_mapes):.2f}% - {np.max(repeated_mapes):.2f}%]")

    print("\n" + "=" * 80)
    print(" METHOD 2: LEAVE-ONE-CONFIG-OUT (STRICT UNSEEN MODEL SPLIT) ")
    print("=" * 80)
    logo = LeaveOneGroupOut()
    group_mapes = []

    for train_idx, val_idx in logo.split(X, y, groups=groups):
        w_fold, _ = nnls(X[train_idx], y[train_idx])
        preds = X[val_idx] @ w_fold
        errs = np.abs(y[val_idx] - preds) / y[val_idx] * 100
        group_mapes.append(np.mean(errs))

    print(f"--> Leave-One-Config-Out Mean MAPE: {np.mean(group_mapes):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
