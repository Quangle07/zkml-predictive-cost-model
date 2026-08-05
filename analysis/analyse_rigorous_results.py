import os
import pandas as pd
import numpy as np

def analyse_results(csv_path="high_res_rigorous_results.csv"):
    # Dynamically resolve path: analysis/../data/filename.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(script_dir), "data", csv_filename)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Could not find {csv_path}. Make sure the benchmarking suite is finished.")
        return
    
    # Calculate the median across the 5 trials
    grouped = df.groupby(['model_type', 'parameter_size']).agg(
        median_time=('proving_time', 'median'),
        std_time=('proving_time', 'std'),
        median_ram=('peak_ram_mb', 'median'),
        logrows=('logrows', 'max')
    ).reset_index()

    print("=" * 70)
    print(" ZKML COST MODEL - HIGH RESOLUTION STATISTICAL ANALYSIS ")
    print("=" * 70)

    # Analyse Dense / Linear Layer (alpha)
    df_lin = grouped[grouped['model_type'] == 'Linear']
    if not df_lin.empty:
        x_lin = df_lin['parameter_size'].values
        y_lin = df_lin['median_time'].values
        slope_lin, intercept_lin = np.polyfit(x_lin, y_lin, 1)
        
        print("\n[1] DENSE / LINEAR LAYER FIT")
        print(f"    - Alpha (Time per output element): {slope_lin * 1000:.4f} ms/element")
        print(f"    - Base Setup Time (R_base):        {intercept_lin:.2f} s")

    # Analyse Sigmoid Lookup Table (beta)
    df_sig = grouped[grouped['model_type'] == 'Sigmoid']
    if not df_sig.empty:
        x_sig = df_sig['parameter_size'].values
        y_sig = df_sig['median_time'].values
        slope_sig, intercept_sig = np.polyfit(x_sig, y_sig, 1)
        
        print("\n[2] SIGMOID (LOOKUP TABLE) FIT")
        print(f"    - Beta (Time per LUT element):     {slope_sig * 1000:.4f} ms/element")
        print(f"    - Base Setup Time (R_base):        {intercept_sig:.2f} s")

    # Analyse Spatial Convolution (gamma)
    df_conv = grouped[grouped['model_type'] == 'Conv2D']
    if not df_conv.empty:
        c_out = df_conv['parameter_size'].values
        # MACs = H_out(16) * W_out(16) * C_in(3) * C_out * K_h(3) * K_w(3)
        macs_conv = 16 * 16 * 3 * c_out * 3 * 3
        y_conv = df_conv['median_time'].values
        slope_conv, intercept_conv = np.polyfit(macs_conv, y_conv, 1)
        
        print("\n[3] SPATIAL CONVOLUTION (CONV2D) FIT")
        print(f"    - Gamma (Time per MAC):            {slope_conv * 1e6:.4f} μs/MAC")
        print(f"    - Base Setup Time (R_base):        {intercept_conv:.2f} s")

    # Full CNN Predictive Validation
    df_cnn = grouped[grouped['model_type'] == 'MiniCNN']
    if not df_cnn.empty:
        print("\n[4] COMPOSITE CNN PREDICTION ACCURACY")
        print(f"    {'Channels':<10} | {'Actual Time':<15} | {'Predicted Time':<15} | {'Error (%)'}")
        print("    " + "-"*60)
        
        mapes = []
        for _, row in df_cnn.iterrows():
            c = row['parameter_size']
            actual = row['median_time']
            
            # Predict the operations
            conv_macs = 16 * 16 * 3 * c * 3 * 3
            lin_ops = (c * 8 * 8) * 10
            
            t_conv = slope_conv * conv_macs
            t_relu = 0.0001 * (c * 16 * 16) # Minimal piecewise overhead
            t_pool = 0.0001 * (c * 8 * 8)   # Minimal pooling overhead
            t_lin = slope_lin * lin_ops
            
            # The model assumes compiler shares the base setup grid cost
            predicted = max(intercept_conv, intercept_lin) + t_conv + t_relu + t_pool + t_lin
            error = abs(actual - predicted) / actual * 100
            mapes.append(error)
            
            print(f"    {c:<10} | {actual:<12.2f} s | {predicted:<12.2f} s | {error:.1f}%")

        print(f"\n    --> Overall Mean Absolute Percentage Error (MAPE): {np.mean(mapes):.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    analyse_results()

#======================================================================
# ZKML COST MODEL - HIGH RESOLUTION STATISTICAL ANALYSIS 
#======================================================================
#
#[1] DENSE / LINEAR LAYER FIT
#    - Alpha (Time per output element): 9.8239 ms/element
#    - Base Setup Time (R_base):        11.81 s
#
#[2] SIGMOID (LOOKUP TABLE) FIT
#    - Beta (Time per LUT element):     3.4271 ms/element
#    - Base Setup Time (R_base):        19.46 s
#
#[3] SPATIAL CONVOLUTION (CONV2D) FIT
#    - Gamma (Time per MAC):            328.2724 μs/MAC
#    - Base Setup Time (R_base):        17.13 s
#
#[4] COMPOSITE CNN PREDICTION ACCURACY
#    Channels   | Actual Time     | Predicted Time  | Error (%)
#    ------------------------------------------------------------
#    8          | 62.27        s | 85.83        s | 37.8%
#    16         | 108.87       s | 154.54       s | 41.9%
#    24         | 153.17       s | 223.24       s | 45.8%
#    32         | 218.56       s | 291.95       s | 33.6%
#    40         | 249.30       s | 360.66       s | 44.7%
#    48         | 295.15       s | 429.36       s | 45.5%
#    56         | 339.49       s | 498.07       s | 46.7%
#    64         | 396.51       s | 566.78       s | 42.9%
#
#    --> Overall Mean Absolute Percentage Error (MAPE): 42.36%
#======================================================================


