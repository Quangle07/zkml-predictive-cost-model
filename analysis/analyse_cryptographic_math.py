import pandas as pd
import numpy as np
from scipy.optimize import nnls
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
                    "actual_proving_time": t,
                    "fft_work": fft_work,
                    "msm_work": float(d_size),
                    "total_assignments": feat["total_assignments"],
                    "lookup_span": feat["lookup_span"],
                    "total_const_size": feat["total_const_size"],
                    "source_file": f
                })

    df_all = pd.DataFrame(dataset_rows).drop_duplicates(subset=["model_type", "size", "actual_proving_time"])

    # Train the 5-term model
    feature_cols = ["fft_work", "msm_work", "total_assignments", "lookup_span", "total_const_size"]
    X = df_all[feature_cols].values
    y = df_all["actual_proving_time"].values
    w, _ = nnls(X, y)

    # Compute predictions
    df_all["predicted_time"] = X @ w
    df_all["abs_error_sec"] = np.abs(df_all["actual_proving_time"] - df_all["predicted_time"])
    df_all["error_pct"] = (df_all["abs_error_sec"] / df_all["actual_proving_time"]) * 100

    print("=" * 100)
    print(" DETAILED ITEMIZATION: ACTUAL VS PREDICTED PROVING TIMES ")
    print("=" * 100)
    print(f"{'Model Architecture':<20} | {'Size':<6} | {'Actual (s)':<10} | {'Pred (s)':<10} | {'Diff (s)':<10} | {'Error (%)'}")
    print("-" * 100)

    # Print sorted by Model Family and Size
    df_sorted = df_all.sort_values(by=["model_type", "size"])
    for _, row in df_sorted.iterrows():
        m_type = row["model_type"]
        sz = row["size"]
        act = row["actual_proving_time"]
        pred = row["predicted_time"]
        diff = row["abs_error_sec"]
        err = row["error_pct"]
        print(f"{m_type:<20} | {sz:<6} | {act:<10.2f} | {pred:<10.2f} | {diff:<10.2f} | {err:.2f}%")

    print("-" * 100)

    # Grouped Summary by Model Family
    print("\n" + "=" * 100)
    print(" ACCURACY SUMMARY BY ARCHITECTURE FAMILY ")
    print("=" * 100)
    for family, group in df_all.groupby("model_type"):
        print(f" - {family:<20} (N={len(group):<2}) | Mean Error: {group['error_pct'].mean():.2f}% | Max Error: {group['error_pct'].max():.2f}%")
    print("=" * 100)

if __name__ == "__main__":
    main()
