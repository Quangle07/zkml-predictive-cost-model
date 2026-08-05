import pandas as pd
import numpy as np

def analyse_blind_test(csv_path="blind_test_results.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Could not find {csv_path}. Make sure the blind test finished.")
        return
    
    # Filter out cluster noise by taking the median of the 3 trials
    grouped = df.groupby(['model_type', 'parameter_size']).agg(
        median_time=('proving_time', 'median')
    ).reset_index()
    
    # Hardcode the scientifically derived coefficients
    # CNN Parameters
    r_base_cnn = 17.13
    gamma = 328.2724e-6
    alpha = 9.8239e-3
    delta_cnn = 0.6786
    
    # Transformer Parameters
    r_base_trans = 123.38
    beta_gelu = 16.1851e-3
    beta_norm = 333.5424e-3
    beta_attn = 2066.6735e-3
    delta_trans = 0.3139
    seq_len = 16

    print("=" * 80)
    print(" ZKML COST MODEL - BLIND TEST VALIDATION ")
    print("=" * 80)
    print(f"{'Model Type':<18} | {'Size':<6} | {'Actual (s)':<12} | {'Predicted (s)':<15} | {'Error (%)'}")
    print("-" * 80)
    
    mapes = []
    for _, row in grouped.iterrows():
        m_type = row['model_type']
        size = row['parameter_size']
        actual = row['median_time']
        
        predicted = 0
        if m_type == "MiniCNN":
            c = size
            # Calculate raw math operations
            conv_macs = 16 * 16 * 3 * c * 3 * 3
            lin_ops = (c * 8 * 8) * 10
            
            # Predict isolated costs
            t_conv = gamma * conv_macs
            t_relu = 0.0001 * (c * 16 * 16)
            t_pool = 0.0001 * (c * 8 * 8)
            t_lin = alpha * lin_ops
            
            sum_parts = t_conv + t_relu + t_pool + t_lin
            
            # Apply base cost and fusion discount
            predicted = r_base_cnn + (delta_cnn * sum_parts)
            
        elif m_type == "TransformerBlock":
            dim = size
            # Predict isolated costs
            t_attn = beta_attn * dim
            t_norm = (beta_norm * dim) * 2
            t_gelu = beta_gelu * (dim * seq_len)
            
            # Calculate massive MLP dense matrix ops
            lin_ops_1 = seq_len * dim * (dim * 4)
            lin_ops_2 = seq_len * (dim * 4) * dim
            t_lin = alpha * (lin_ops_1 + lin_ops_2)
            
            sum_parts = t_attn + t_norm + t_gelu + t_lin
            
            # Apply base cost and fusion discount
            predicted = r_base_trans + (delta_trans * sum_parts)
        
        # Calculate accuracy
        error = abs(actual - predicted) / actual * 100
        mapes.append(error)
        
        print(f"{m_type:<18} | {size:<6} | {actual:<12.2f} | {predicted:<15.2f} | {error:.2f}%")
        
    print("-" * 80)
    print(f"--> Overall Blind Test Mean Absolute Percentage Error (MAPE): {np.mean(mapes):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    analyse_blind_test()
