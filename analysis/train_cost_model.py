import pandas as pd
import numpy as np
import os

def main():
    # Data
    csv_file = "circuit_features_master.csv"

    if not os.path.exists(csv_file):
        print(f"[!] Error: Could not find {csv_file}")
        print("Make sure you are running this in the same directory as your data.")
        return

    df = pd.read_csv(csv_file)

    print(f"Loaded {len(df)} configurations from {csv_file}...\n")

    # Setup the Regression Matrices
    # X represents the independent circuit features
    X = df[['domain_fft_work', 'total_assignments', 'lookup_span']].values

    # We append a column of ones to the X matrix to solve for the intercept (beta_0)
    X_design = np.hstack([X, np.ones((X.shape[0], 1))])

    # y is our dependent variable (the actual measured time)
    y = df['actual_proving_time'].values

    # Solve for the Coefficients
    # np.linalg.lstsq solves the equation y = X * beta using Ordinary Least Squares
    coefs, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)

    # Extract the Results
    print("==================================================")
    print("    CIRCUIT-LEVEL COST MODEL REGRESSION RESULTS   ")
    print("==================================================")
    print(f"FFT Work Coefficient (beta_1):     {coefs[0]:.6e}")
    print(f"Assignments Coefficient (beta_2):  {coefs[1]:.6e}")
    print(f"Lookup Span Coefficient (beta_3):  {coefs[2]:.6e}")
    print(f"Intercept / Base Time (beta_0):    {coefs[3]:.2f} s")
    print("==================================================")

if __name__ == "__main__":
    main()
