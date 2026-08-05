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
    
    # Strip the noise: Calculate the median across the 5 trials
    grouped = df.groupby(['model_type', 'parameter_size']).agg(
        median_time=('proving_time', 'median'),
        std_time=('proving_time', 'std'),
        median_ram=('peak_ram_mb', 'median'),
        logrows=('logrows', 'max')
    ).reset_index()

    print("=" * 75)
    print(" ZKML COST MODEL - HIGH RESOLUTION STATISTICAL ANALYSIS ")
    print("=" * 75)

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
        macs_conv = 16 * 16 * 3 * c_out * 3 * 3
        y_conv = df_conv['median_time'].values
        slope_conv, intercept_conv = np.polyfit(macs_conv, y_conv, 1)
        
        print("\n[3] SPATIAL CONVOLUTION (CONV2D) FIT")
        print(f"    - Gamma (Time per MAC):            {slope_conv * 1e6:.4f} μs/MAC")
        print(f"    - Base Setup Time (R_base):        {intercept_conv:.2f} s")

    # Full CNN Predictive Validation with Delta
    df_cnn = grouped[grouped['model_type'] == 'MiniCNN']
    if not df_cnn.empty:
        # Base setup is shared, so we take the max of the intercepts
        r_base = max(intercept_conv, intercept_lin)
        
        # PASS 1: Calculate the Delta (Fusion Discount)
        deltas = []
        for _, row in df_cnn.iterrows():
            c = row['parameter_size']
            actual = row['median_time']
            
            # Predict the raw operations
            conv_macs = 16 * 16 * 3 * c * 3 * 3
            lin_ops = (c * 8 * 8) * 10
            
            t_conv = slope_conv * conv_macs
            t_relu = 0.0001 * (c * 16 * 16) 
            t_pool = 0.0001 * (c * 8 * 8)   
            t_lin = slope_lin * lin_ops
            
            sum_of_parts = t_conv + t_relu + t_pool + t_lin
            actual_ops_cost = actual - r_base
            
            delta = actual_ops_cost / sum_of_parts
            deltas.append(delta)
            
        mean_delta = np.mean(deltas)
        
        print(f"\n[4] COMPILER FUSION DISCOUNT (DELTA)")
        print(f"    - Calculated Delta:                {mean_delta:.4f}")
        print(f"    - Cost Formula:                    Cost = {r_base:.2f}s + {mean_delta:.2f} * SUM(Layer Costs)")

        # PASS 2: Calculate New Prediction Accuracy
        print("\n[5] COMPOSITE CNN PREDICTION ACCURACY (WITH DELTA)")
        print(f"    {'Channels':<10} | {'Actual Time':<15} | {'Predicted Time':<15} | {'Error (%)'}")
        print("    " + "-"*65)
        
        mapes = []
        for _, row in df_cnn.iterrows():
            c = row['parameter_size']
            actual = row['median_time']
            
            conv_macs = 16 * 16 * 3 * c * 3 * 3
            lin_ops = (c * 8 * 8) * 10
            
            t_conv = slope_conv * conv_macs
            t_relu = 0.0001 * (c * 16 * 16)
            t_pool = 0.0001 * (c * 8 * 8)
            t_lin = slope_lin * lin_ops
            
            sum_of_parts = t_conv + t_relu + t_pool + t_lin
            
            # Apply the new delta formula
            predicted = r_base + (mean_delta * sum_of_parts)
            
            error = abs(actual - predicted) / actual * 100
            mapes.append(error)
            
            print(f"    {c:<10} | {actual:<12.2f} s | {predicted:<12.2f} s | {error:.1f}%")

        print(f"\n    --> Overall Mean Absolute Percentage Error (MAPE): {np.mean(mapes):.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    analyse_results()

#===========================================================================
# ZKML COST MODEL - HIGH RESOLUTION STATISTICAL ANALYSIS 
#===========================================================================
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
#[4] COMPILER FUSION DISCOUNT (DELTA)
#    - Calculated Delta:                0.6786
#    - Cost Formula:                    Cost = 17.13s + 0.68 * SUM(Layer Costs)
#
#[5] COMPOSITE CNN PREDICTION ACCURACY (WITH DELTA)
#    Channels   | Actual Time     | Predicted Time  | Error (%)
#    -----------------------------------------------------------------
#    8          | 62.27        s | 63.75        s | 2.4%
#    16         | 108.87       s | 110.37       s | 1.4%
#    24         | 153.17       s | 156.99       s | 2.5%
#    32         | 218.56       s | 203.61       s | 6.8%
#    40         | 249.30       s | 250.23       s | 0.4%
#    48         | 295.15       s | 296.86       s | 0.6%
#    56         | 339.49       s | 343.48       s | 1.2%
#    64         | 396.51       s | 390.10       s | 1.6%
#
#    --> Overall Mean Absolute Percentage Error (MAPE): 2.10%
#===========================================================================

