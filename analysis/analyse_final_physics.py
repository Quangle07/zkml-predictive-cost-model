import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os

def test_on_dataset(file_path, w_dom, w_assign, w_lookup, w_const, df_features):
    if not os.path.exists(file_path):
        print(f"[!] Skipping {file_path} (File not found)")
        return

    df_test = pd.read_csv(file_path)
    print(f"\n{'=' * 80}")
    print(f" TESTING ON: {file_path} ")
    print(f"{'=' * 80}")

    # Standardise Time column
    time_col = next((c for c in ["actual_proving_time", "proving_time", "prove_time", "time"] if c in df_test.columns), None)
    if not time_col:
        print(f"[!] Could not find a proving time column in {file_path}. Skipping.")
        return

    # Standardise Model/Size columns for merging
    if "model_type" not in df_test.columns and "model" in df_test.columns:
        df_test = df_test.rename(columns={"model": "model_type"})
    if "parameter_size" not in df_test.columns and "size" in df_test.columns:
        df_test = df_test.rename(columns={"size": "parameter_size"})

    # MERGE: Map circuit features from the master list if they are missing
    if "total_assignments" not in df_test.columns:
        # Extract just the unique features from the master dataset
        df_feat_unique = df_features[['model_type', 'parameter_size', 'domain_size',
                                      'total_assignments', 'lookup_span', 'total_const_size']].drop_duplicates()

        # SQL-style Left Join on Model and Size
        df_test = pd.merge(df_test, df_feat_unique, on=['model_type', 'parameter_size'], how='left')

    print(f"{'Model Type':<18} | {'Size':<6} | {'Actual (s)':<12} | {'Predicted (s)':<15} | {'Error (%)'}")
    print("-" * 80)

    mapes = []
    for _, row in df_test.iterrows():
        m_type = row["model_type"]
        size = int(row["parameter_size"])
        actual_time = row[time_col]

        # Skip if the merge couldn't find features for this specific model size
        if pd.isna(row.get("total_assignments")):
            continue

        # Calculate purely based on circuit features
        pred_time = (row["domain_size"] * w_dom +
                     row["total_assignments"] * w_assign +
                     row["lookup_span"] * w_lookup +
                     row["total_const_size"] * w_const)

        error = abs(actual_time - pred_time) / actual_time * 100
        mapes.append(error)

        print(f"{m_type:<18} | {size:<6} | {actual_time:<12.2f} | {pred_time:<15.2f} | {error:.2f}%")

    if mapes:
        print("-" * 80)
        print(f"--> Overall MAPE for {file_path}: {np.mean(mapes):.2f}%")
    print("=" * 80)

def main():
    if not os.path.exists("circuit_features_master.csv"):
        print("[!] Missing circuit_features_master.csv to train the model.")
        return

    # Load Training Data
    df_train = pd.read_csv("circuit_features_master.csv")
    features = ["domain_size", "total_assignments", "lookup_span", "total_const_size"]
    X_train = df_train[features].values

    # Train Pure NNLS
    time_col_train = next((c for c in ["actual_proving_time", "proving_time", "prove_time"] if c in df_train.columns), None)
    y_train = df_train[time_col_train].values

    w_actual, _ = nnls(X_train, y_train)
    w_dom, w_assign, w_lookup, w_const = w_actual

    print("=" * 80)
    print(" THE PURE PHYSICS COST MODEL (ZERO-INTERCEPT NNLS) ")
    print("=" * 80)
    print(f"Fitted Model Coefficients:")
    print(f"  - Grid Bounds (Domain Size): {w_dom:.6e}")
    print(f"  - Grid Density (Assigns):    {w_assign:.6e}")
    print(f"  - Weights (Const Size):      {w_const:.6e}")
    print(f"  - Non-Linearity (Lookup):    {w_lookup:.6e}")

    # Test on all available datasets, passing df_train so it can map missing features
    test_on_dataset("deep_validation_results.csv", w_dom, w_assign, w_lookup, w_const, df_train)
    test_on_dataset("high_res_rigorous_results.csv", w_dom, w_assign, w_lookup, w_const, df_train)
    test_on_dataset("transformer_block_results.csv", w_dom, w_assign, w_lookup, w_const, df_train)

if __name__ == "__main__":
    main()
