import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

def main():
    if not os.path.exists("circuit_features_master.csv") or not os.path.exists("deep_validation_results.csv"):
        print("[!] Missing required CSV files.")
        return

    # Load Training Data
    df_train = pd.read_csv("circuit_features_master.csv")
    features = ["domain_size", "total_assignments", "lookup_span", "total_const_size"]

    X_train = df_train[features].values
    y_train = df_train["actual_proving_time"].values

    # DOUBLE NORMALISATION FIX
    X_max = X_train.max(axis=0)
    y_max = y_train.max()

    X_train_norm = X_train / X_max
    y_train_norm = y_train / y_max  # Normalise the target variable too!

    # Define the Ridge Loss Function (NO INTERCEPT)
    alpha = 0.5  # Lowered regularisation for properly scaled data

    def loss_fn(w):
        preds = X_train_norm @ w
        mse = np.mean((y_train_norm - preds)**2)
        l2 = alpha * np.sum(w**2)
        return mse + l2

    # Force Physics (Bounds >= 0)
    init_guess = np.ones(len(features)) / len(features)
    bounds = [(0, None) for _ in range(len(features))]

    # Optimise
    res = minimize(loss_fn, init_guess, bounds=bounds, method='L-BFGS-B')

    # De-normalise weights back to actual seconds
    # Formula: w_actual = (w_norm * y_max) / X_max
    w_actual = (res.x * y_max) / X_max
    w_dom, w_assign, w_lookup, w_const = w_actual

    print("=" * 80)
    print(" TRUE HYBRID COST MODEL (PROPERLY SCALED ZERO-INTERCEPT) ")
    print("=" * 80)
    print(f"Fitted Model Coefficients:")
    print(f"  - Grid Bounds (Domain Size): {w_dom:.6e}  <-- The Step Base")
    print(f"  - Grid Density (Assigns):    {w_assign:.6e}  <-- The Slope")
    print(f"  - Weights (Const Size):      {w_const:.6e}  <-- The Slope")
    print(f"  - Non-Linearity (Lookup):    {w_lookup:.6e}  <-- The Spike")
    print("-" * 80)

    # Test on Deep Validation Data
    df_test = pd.read_csv("deep_validation_results.csv")

    print(f"{'Model Type':<18} | {'Size':<6} | {'Actual (s)':<12} | {'Predicted (s)':<15} | {'Error (%)'}")
    print("-" * 80)

    mapes = []
    for _, row in df_test.iterrows():
        m_type = row["model_type"]
        size = int(row["parameter_size"])
        actual_time = row["actual_proving_time"]

        # Calculate purely based on circuit features (Intercept is 0)
        pred_time = (row["domain_size"] * w_dom +
                     row["total_assignments"] * w_assign +
                     row["lookup_span"] * w_lookup +
                     row["total_const_size"] * w_const)

        error = abs(actual_time - pred_time) / actual_time * 100
        mapes.append(error)

        print(f"{m_type:<18} | {size:<6} | {actual_time:<12.2f} | {pred_time:<15.2f} | {error:.2f}%")

    print("-" * 80)
    print(f"--> True Hybrid Overall MAPE: {np.mean(mapes):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
