import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os

def main():
    # Build a "Feature Dictionary" from the files that contain the deep circuit features
    feature_dict = {}
    feature_files = ['circuit_features_master.csv', 'deep_validation_results.csv']

    for f in feature_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                m = row.get("model_type", row.get("model"))
                s = int(row.get("parameter_size", row.get("size", 0)))
                # Only grab rows that actually contain the extracted assignments
                if "total_assignments" in df.columns and not pd.isna(row["total_assignments"]):
                    feature_dict[(m, s)] = {
                        "domain_size": row.get("domain_size", 0),
                        "total_assignments": row["total_assignments"],
                        "lookup_span": row.get("lookup_span", 0),
                        "total_const_size": row.get("total_const_size", 0)
                    }

    # Gather ALL benchmarking runs and map the features to them
    all_files = [
        'circuit_features_master.csv',
        'deep_validation_results.csv',
        'high_res_rigorous_results.csv',
        'transformer_block_results.csv'
    ]

    dataset_rows = []
    for f in all_files:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)

        # Standardise time column
        time_col = next((c for c in ["actual_proving_time", "proving_time", "prove_time", "time"] if c in df.columns), None)
        if not time_col:
            continue

        for _, row in df.iterrows():
            m = row.get("model_type", row.get("model"))
            s = int(row.get("parameter_size", row.get("size", 0)))
            t = row[time_col]

            if pd.isna(t):
                continue

            # If we know the circuit physics for this architecture/size, add it to our master dataset
            if (m, s) in feature_dict:
                feat = feature_dict[(m, s)]
                dataset_rows.append({
                    "model_type": m,
                    "size": s,
                    "actual_proving_time": t,
                    "domain_size": feat["domain_size"],
                    "total_assignments": feat["total_assignments"],
                    "lookup_span": feat["lookup_span"],
                    "total_const_size": feat["total_const_size"],
                    "source_file": f
                })

    df_all = pd.DataFrame(dataset_rows)
    if df_all.empty:
        print("[!] No valid data successfully merged.")
        return

    print(f"[*] Successfully merged {len(df_all)} total benchmarks across all files.")

    # Shuffle the dataset randomly (Seed 42 for reproducibility)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    # Perform 5-Fold Cross-Validation
    k = 5
    fold_size = len(df_all) // k
    features = ["domain_size", "total_assignments", "lookup_span", "total_const_size"]

    fold_mapes = []
    fold_coefs = []

    print("\n" + "="*80)
    print(" 5-FOLD CROSS-VALIDATION (ZERO-INTERCEPT PURE PHYSICS)")
    print("="*80)

    for i in range(k):
        # Slice out the unseen validation chunk
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(df_all)

        val_idx = list(range(start, end))
        train_idx = [idx for idx in range(len(df_all)) if idx not in val_idx]

        df_train = df_all.iloc[train_idx]
        df_val = df_all.iloc[val_idx]

        X_train = df_train[features].values
        y_train = df_train["actual_proving_time"].values

        X_val = df_val[features].values
        y_val = df_val["actual_proving_time"].values

        # Train on 80%
        w_fold, _ = nnls(X_train, y_train)
        fold_coefs.append(w_fold)

        # Evaluate on the 20% unseen data
        preds = X_val @ w_fold
        errors = np.abs(y_val - preds) / y_val * 100
        fold_mape = np.mean(errors)
        fold_mapes.append(fold_mape)

        print(f"Fold {i+1}: MAPE = {fold_mape:>5.2f}% | Train N={len(df_train)} | Test N={len(df_val)}")
        print(f"  -> Coefs: D_size:{w_fold[0]:.2e}, Assigns:{w_fold[1]:.2e}, Lookup:{w_fold[2]:.2e}, Const:{w_fold[3]:.2e}")
        print("-" * 80)

    # Summarise the Final Average Results
    avg_mape = np.mean(fold_mapes)
    avg_coefs = np.mean(fold_coefs, axis=0)

    print("=" * 80)
    print(" 5-FOLD CROSS-VALIDATION SUMMARY ")
    print("=" * 80)
    print(f"Average Out-of-Sample Error (MAPE): {avg_mape:.2f}%")
    print("\nFinal Stabilized Formula Coefficients:")
    print(f"  - Grid Bounds (Domain Size): {avg_coefs[0]:.6e}")
    print(f"  - Grid Density (Assigns):    {avg_coefs[1]:.6e}")
    print(f"  - Non-Linearity (Lookup):    {avg_coefs[2]:.6e}")
    print(f"  - Weights (Const Size):      {avg_coefs[3]:.6e}")
    print("=" * 80)

if __name__ == "__main__":
    main()
