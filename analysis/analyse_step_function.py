import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os

def main():
    if not os.path.exists("circuit_features_master.csv") or not os.path.exists("deep_validation_results.csv"):
        print("[!] Missing required CSV files.")
        return

    # Load Training Data (Original Master Dataset)
    df_train = pd.read_csv("circuit_features_master.csv")

    # Force the model to use ONLY step-function features and lookup penalties
    feature_cols = [
        "domain_fft_work",  # The O(n log n) step function
        "domain_size",      # The O(n) step function
        "lookup_span"       # The non-linear penalty
    ]

    X_train = df_train[feature_cols].values
    X_design = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    y_train = df_train["actual_proving_time"].values

    # Train the Constrained NNLS model
    coefs, _ = nnls(X_design, y_train)
    w_fft, w_dom, w_lookup, intercept = coefs

    print("=" * 80)
    print(" GRID-AWARE COST MODEL (THEORETICAL PLONKISH BOUNDS) ")
    print("=" * 80)
    print(f"Fitted Model Coefficients:")
    print(f"  - FFT Work (n log n): {w_fft:.6e}")
    print(f"  - Domain Size (n):    {w_dom:.6e}")
    print(f"  - Lookup Span:        {w_lookup:.6e}")
    print(f"  - Intercept (R_base): {intercept:.2f} s")
    print("-" * 80)

    # Load Testing Data (Deep Validation Dataset)
    df_test = pd.read_csv("deep_validation_results.csv")

    print(f"{'Model Type':<18} | {'Size':<6} | {'Actual (s)':<12} | {'Predicted (s)':<15} | {'Error (%)'}")
    print("-" * 80)

    mapes = []
    for _, row in df_test.iterrows():
        m_type = row["model_type"]
        size = int(row["parameter_size"])
        actual_time = row["actual_proving_time"]

        # Calculate using the strict step function formula
        pred_time = (row["domain_fft_work"] * w_fft +
                     row["domain_size"] * w_dom +
                     row["lookup_span"] * w_lookup +
                     intercept)

        error = abs(actual_time - pred_time) / actual_time * 100
        mapes.append(error)

        print(f"{m_type:<18} | {size:<6} | {actual_time:<12.2f} | {pred_time:<15.2f} | {error:.2f}%")

    print("-" * 80)
    print(f"--> Grid-Aware Overall MAPE: {np.mean(mapes):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
