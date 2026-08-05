import pandas as pd
import numpy as np

def analyse_transformers():
    # Find the absolute path to the data/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the path to the analysis/ folder
    data_dir = os.path.join(script_dir, '..', 'data')       # Goes up one level, then into data/

    # Construct the full file paths
    iso_path = os.path.join(data_dir, "transformer_benchmark_results.csv")
    block_path = os.path.join(data_dir, "transformer_block_results.csv")
    cnn_path = os.path.join(data_dir, "high_res_rigorous_results.csv")
    
    # Load all datasets
    try:
        df_iso = pd.read_csv("transformer_benchmark_results.csv")
        df_block = pd.read_csv("transformer_block_results.csv")
        df_cnn = pd.read_csv("high_res_rigorous_results.csv") # We need this for the Linear coefficient
    except FileNotFoundError as e:
        print(f"[!] Missing data file: {e}")
        return

    # Get Medians
    iso_grouped = df_iso.groupby(['model_type', 'parameter_size']).agg(median_time=('proving_time', 'median')).reset_index()
    block_grouped = df_block.groupby(['model_type', 'parameter_size']).agg(median_time=('proving_time', 'median')).reset_index()
    cnn_grouped = df_cnn.groupby(['model_type', 'parameter_size']).agg(median_time=('proving_time', 'median')).reset_index()

    print("=" * 75)
    print(" ZKML COST MODEL - TRANSFORMER ARCHITECTURE ANALYSIS ")
    print("=" * 75)

    # Extract Component Costs
    df_gelu = iso_grouped[iso_grouped['model_type'] == 'GELU']
    slope_gelu, int_gelu = np.polyfit(df_gelu['parameter_size'].values, df_gelu['median_time'].values, 1)

    df_norm = iso_grouped[iso_grouped['model_type'] == 'LayerNorm']
    slope_norm, int_norm = np.polyfit(df_norm['parameter_size'].values, df_norm['median_time'].values, 1)

    df_attn = iso_grouped[iso_grouped['model_type'] == 'SelfAttention']
    slope_attn, int_attn = np.polyfit(df_attn['parameter_size'].values, df_attn['median_time'].values, 1)

    # Extract Linear cost from the rigorous suite
    df_lin = cnn_grouped[cnn_grouped['model_type'] == 'Linear']
    slope_lin, int_lin = np.polyfit(df_lin['parameter_size'].values, df_lin['median_time'].values, 1)

    print("\n[1] ISOLATED TRANSFORMER COMPONENTS")
    print(f"    - Linear    (Time per element): {slope_lin * 1000:.4f} ms/element")
    print(f"    - GELU      (Time per element): {slope_gelu * 1000:.4f} ms/element")
    print(f"    - LayerNorm (Time per dim):     {slope_norm * 1000:.4f} ms/dim")
    print(f"    - Attention (Time per dim):     {slope_attn * 1000:.4f} ms/dim")
    
    r_base_trans = max(int_gelu, int_norm, int_attn, int_lin)
    print(f"    - Shared Base Setup (R_base):   {r_base_trans:.2f} s")

    # Predict the Full Block & Calculate Fusion Delta
    print("\n[2] TRANSFORMER BLOCK FUSION TEST (FIXED)")
    print(f"    {'Dim':<6} | {'Actual Time':<12} | {'Sum of Parts':<15} | {'Implied Delta'}")
    print("    " + "-"*55)
    
    deltas = []
    
    for _, row in block_grouped.iterrows():
        dim = row['parameter_size']
        actual = row['median_time']
        
        seq_len = 16
        
        # Core components
        t_attn = slope_attn * dim
        t_norm = (slope_norm * dim) * 2
        t_gelu = slope_gelu * (dim * seq_len)
        
        # ADDING THE MISSING LINEAR LAYERS:
        # A Transformer block Feed-Forward Network projects `dim` -> `dim*4`, then `dim*4` -> `dim`.
        # Across a sequence of length 16, the total multiply-accumulate operations are:
        lin_ops_1 = seq_len * dim * (dim * 4)
        lin_ops_2 = seq_len * (dim * 4) * dim
        total_lin_ops = lin_ops_1 + lin_ops_2
        
        t_lin = slope_lin * total_lin_ops
        
        # Now we sum ALL parts properly
        sum_of_parts = t_attn + t_norm + t_gelu + t_lin
        
        # Calculate Delta
        actual_ops = actual - r_base_trans
        delta = actual_ops / sum_of_parts
        deltas.append(delta)
        
        print(f"    {dim:<6} | {actual:<10.2f} s | {sum_of_parts:<13.2f} s | {delta:.4f}")

    print(f"\n    --> Mean Transformer Fusion Delta: {np.mean(deltas):.4f}")
    print("=" * 75)

if __name__ == "__main__":
    analyse_transformers()

#===========================================================================
# ZKML COST MODEL - TRANSFORMER ARCHITECTURE ANALYSIS 
#===========================================================================
#
#[1] ISOLATED TRANSFORMER COMPONENTS
#    - Linear    (Time per element): 9.8239 ms/element
#    - GELU      (Time per element): 16.1851 ms/element
#    - LayerNorm (Time per dim):     333.5424 ms/dim
#    - Attention (Time per dim):     2066.6735 ms/dim
#    - Shared Base Setup (R_base):   123.38 s
#
#[2] TRANSFORMER BLOCK FUSION TEST (FIXED)
#    Dim    | Actual Time  | Sum of Parts    | Implied Delta
#    -------------------------------------------------------
#    16     | 276.73     s | 369.79        s | 0.4147
#    32     | 417.83     s | 1383.41       s | 0.2128
#
#    --> Mean Transformer Fusion Delta: 0.3138
#===========================================================================
