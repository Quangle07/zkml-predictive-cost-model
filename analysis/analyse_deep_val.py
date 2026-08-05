import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os

def main():
    if not os.path.exists("circuit_features_master.csv") or not os.path.exists("deep_validation_results.csv"):
        print("[!] Missing required CSV files. Is the cluster job finished?")
        return

    # 1Load Training Data (Original Master Dataset)
    df_train = pd.read_csv("circuit_features_master.csv")
    feature_cols = [
        "domain_fft_work", "domain_size", "num_rows",
        "total_assignments", "lookup_span", "num_shuffles", "total_const_size"
    ]

    X_train = df_train[feature_cols].values
    X_design = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    y_train = df_train["actual_proving_time"].values

    # Train the Constrained NNLS model
    coefs, _ = nnls(X_design, y_train)
    w_fft, w_dom, w_rows, w_assign, w_lookup, w_shuf, w_const, intercept = coefs

    print("=" * 80)
    print(" DEEP VALIDATION: NNLS COST MODEL RESULTS ")
    print("=" * 80)
    print(f"Fitted Model Coefficients (Trained on Master Data):")
    print(f"  - FFT Work:           {w_fft:.6e}")
    print(f"  - Domain Size:        {w_dom:.6e}")
    print(f"  - Num Rows:           {w_rows:.6e}")
    print(f"  - Assignments:        {w_assign:.6e}")
    print(f"  - Lookup Span:        {w_lookup:.6e}")
    print(f"  - Shuffles:           {w_shuf:.6e}")
    print(f"  - Constants Size:     {w_const:.6e}")
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

        pred_time = (row["domain_fft_work"] * w_fft +
                     row["domain_size"] * w_dom +
                     row["num_rows"] * w_rows +
                     row["total_assignments"] * w_assign +
                     row["lookup_span"] * w_lookup +
                     row["num_shuffles"] * w_shuf +
                     row["total_const_size"] * w_const +
                     intercept)

        error = abs(actual_time - pred_time) / actual_time * 100
        mapes.append(error)

        print(f"{m_type:<18} | {size:<6} | {actual_time:<12.2f} | {pred_time:<15.2f} | {error:.2f}%")

    print("-" * 80)
    print(f"--> Deep Validation Overall MAPE: {np.mean(mapes):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
