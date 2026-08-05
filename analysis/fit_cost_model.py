import os
import pandas as pd
import numpy as np

def load_dataset(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go into the "data" folder
    data_dir = os.path.join(script_dir, "..", "data")
    # Combine them to get the exact file path
    file_path = os.path.join(data_dir, filename)
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"[!] Warning: Data file not found at {os.path.abspath(file_path)}")
        return None

# Load Datasets
df_linear = load_dataset("linear_only_results.csv")
df_sigmoid = load_dataset("sigmoid_results.csv")
df_conv2d = load_dataset("conv2d_results.csv")
df_cnn = load_dataset("cnn_results.csv")

print("=" * 60)
print(" ZKML PREDICTIVE COST MODEL FITTING RESULTS ")
print("=" * 60)

# Fit Linear Layer Coefficient (alpha)
if df_linear is not None:
    # Feature: Input_Size * Output_Size (here input=1, output=size)
    x_lin = df_linear["size"].values
    y_time_lin = df_linear["proving_time"].values
    
    # Fit linear slope: Proving Time = alpha * N + base_time
    slope_lin, intercept_lin = np.polyfit(x_lin, y_time_lin, 1)
    print(f"\n[1] DENSE / LINEAR LAYER FIT:")
    print(f"    - Alpha (Time per output element): {slope_lin * 1000:.4f} ms/element")
    print(f"    - Baseline Time Intercept:         {intercept_lin:.2f} s")
else:
    print("\n[!] linear_only_results.csv not found.")

# Fit Sigmoid Lookup Table Coefficient (beta)
if df_sigmoid is not None:
    x_sig = df_sigmoid["size"].values
    y_time_sig = df_sigmoid["proving_time"].values
    
    slope_sig, intercept_sig = np.polyfit(x_sig, y_time_sig, 1)
    print(f"\n[2] SIGMOID (LOOKUP TABLE) FIT:")
    print(f"    - Beta (Time per LUT lookup element): {slope_sig * 1000:.4f} ms/element")
    print(f"    - LUT Table Setup Base Time:          {intercept_sig:.2f} s")
else:
    print("\n[!] sigmoid_results.csv not found.")

# Fit Conv2d Spatial Coefficient (gamma)
if df_conv2d is not None:
    # Conv2d FLOP proxy: H_out * W_out * C_in * C_out * K_h * K_w
    # Input was (1, 3, 16, 16), kernel 3x3, padding 1 -> output (1, channels, 16, 16)
    flops_conv = 16 * 16 * 3 * df_conv2d["out_channels"].values * 3 * 3
    y_time_conv = df_conv2d["proving_time"].values
    
    slope_conv, intercept_conv = np.polyfit(flops_conv, y_time_conv, 1)
    print(f"\n[3] SPATIAL CONVOLUTION (CONV2D) FIT:")
    print(f"    - Gamma (Time per kernel Multiply-Accumulate): {slope_conv * 1e6:.4f} μs/MAC")
    print(f"    - Conv Baseline Base Time:                      {intercept_conv:.2f} s")
else:
    print("\n[!] conv2d_results.csv not found.")

# Composite CNN Validation (Predicting MiniCNN)
if df_cnn is not None and all(df is not None for df in [df_linear, df_sigmoid, df_conv2d]):
    print(f"\n[4] COMPOSITE CNN PIPELINE VALIDATION:")
    print(f"    Evaluating Model Additivity on MiniCNN Pipeline...")
    
    predictions = []
    actuals = df_cnn["proving_time"].values
    
    for _, row in df_cnn.iterrows():
        c = row["channels"]
        
        # Calculate operations for each layer in MiniCNN
        # Layer 1: Conv2d (3 in, c out, 16x16 img, 3x3 kernel)
        conv_macs = 16 * 16 * 3 * c * 3 * 3
        time_conv = slope_conv * conv_macs
        
        # Layer 2: ReLU (c * 16 * 16 elements) - minimal piecewise overhead
        time_relu = 0.0001 * (c * 16 * 16) 
        
        # Layer 3: MaxPool2d (16x16 -> 8x8)
        time_pool = 0.0001 * (c * 8 * 8)
        
        # Layer 4: Linear (c * 8 * 8 in -> 10 out)
        lin_ops = (c * 8 * 8) * 10
        time_lin = slope_lin * lin_ops
        
        # Estimated total = max baseline + layer sum
        estimated_time = max(intercept_conv, intercept_lin) + time_conv + time_relu + time_pool + time_lin
        predictions.append(estimated_time)
    
    predictions = np.array(predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print("\n    Channels | Actual Time | Predicted Time | Error (%)")
    print("    " + "-" * 50)
    for c, act, pred in zip(df_cnn["channels"].values, actuals, predictions):
        err = abs(act - pred) / act * 100
        print(f"    {c:8d} | {act:10.2f}s | {pred:13.2f}s | {err:8.1f}%")
        
    print(f"\n    --> Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

print("=" * 60)

#============================================================
# ZKML PREDICTIVE COST MODEL FITTING RESULTS 
#============================================================
#
#[1] DENSE / LINEAR LAYER FIT:
#    - Alpha (Time per output element): 6.6071 ms/element
#    - Baseline Time Intercept:         26.96 s
#
#[2] SIGMOID (LOOKUP TABLE) FIT:
#    - Beta (Time per LUT lookup element): 2.9196 ms/element
#    - LUT Table Setup Base Time:          28.41 s
#
#[3] SPATIAL CONVOLUTION (CONV2D) FIT:
#    - Gamma (Time per kernel Multiply-Accumulate): 452.2031 μs/MAC
#    - Conv Baseline Base Time:                      27.28 s
#
#[4] COMPOSITE CNN PIPELINE VALIDATION:
#    Evaluating Model Additivity on MiniCNN Pipeline...
#
#    Channels | Actual Time | Predicted Time | Error (%)
#    --------------------------------------------------
#           8 |      75.73s |         86.37s |     14.1%
#          16 |     127.04s |        145.46s |     14.5%
#          24 |     183.95s |        204.55s |     11.2%
#
#    --> Mean Absolute Percentage Error (MAPE): 13.25%
#============================================================
