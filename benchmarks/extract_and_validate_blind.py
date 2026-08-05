import json
import os
import torch
import torch.nn as nn
import ezkl
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# Define Model Architectures
class MiniCNN(nn.Module):
    def __init__(self, out_channels):
        super(MiniCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(out_channels * 8 * 8, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TransformerBlockModel(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(TransformerBlockModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
    def forward(self, x):
        return self.encoder_layer(x)

# Circuit Feature Extraction Routine
def extract_features(settings_path):
    with open(settings_path, "r") as file:
        settings = json.load(file)
    
    run_args = settings["run_args"]
    logrows = run_args["logrows"]
    domain_size = 1 << logrows
    
    lookup_min, lookup_max = run_args["lookup_range"]
    lookup_span = lookup_max - lookup_min
    
    return {
        "domain_fft_work": domain_size * logrows,
        "domain_size": domain_size,
        "num_rows": settings["num_rows"],
        "total_assignments": settings["total_assignments"],
        "lookup_span": lookup_span,
        "num_shuffles": settings.get("num_shuffles", 0),
        "total_const_size": settings["total_const_size"]
    }

def get_blind_features(model, dummy_input, model_name, size):
    prefix = f"blind_feat_{model_name}_{size}"
    onnx_path = f"{prefix}.onnx"
    data_path = f"data_{prefix}.json"
    settings_path = f"settings_{prefix}.json"

    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=14, do_constant_folding=True, dynamo=False)
    with open(data_path, "w") as f:
        json.dump(dict(input_data=[dummy_input.detach().numpy().reshape([-1]).tolist()]), f)

    try:
        ezkl.gen_settings(onnx_path, settings_path)
        ezkl.calibrate_settings(data_path, onnx_path, settings_path, "resources")
        feats = extract_features(settings_path)
        return feats
    except Exception as e:
        print(f"[!] Failed to extract features for {model_name} {size}: {e}")
        return None
    finally:
        for f in [onnx_path, data_path, settings_path]:
            if os.path.exists(f): os.remove(f)

# Main Training & Validation Pipeline
def main():
    if not os.path.exists("circuit_features_master.csv"):
        print("[!] Could not find circuit_features_master.csv")
        return
    
    df_train = pd.read_csv("circuit_features_master.csv")
    
    # 7-Variable Multiple Linear Regression
    feature_cols = ["domain_fft_work", "domain_size", "num_rows", "total_assignments", "lookup_span", "num_shuffles", "total_const_size"]
    
    X_train = df_train[feature_cols].values
    X_design = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    y_train = df_train["actual_proving_time"].values
    
    # Non-Negative Least Squares (NNLS) to prevent "negative time" overfitting
    coefs, rnorm = nnls(X_design, y_train)
    w_fft, w_dom, w_rows, w_assign, w_lookup, w_shuf, w_const, intercept = coefs
    
    print("=" * 80)
    print(" CONSTRAINED (NNLS) COST MODEL - BLIND TEST VALIDATION ")
    print("=" * 80)
    print(f"Fitted Model Coefficients:")
    print(f"  - FFT Work (n log n): {w_fft:.6e}")
    print(f"  - Domain Size:        {w_dom:.6e}")
    print(f"  - Num Rows:           {w_rows:.6e}")
    print(f"  - Assignments:        {w_assign:.6e}")
    print(f"  - Lookup Span:        {w_lookup:.6e}")
    print(f"  - Shuffles:           {w_shuf:.6e}")
    print(f"  - Constants Size:     {w_const:.6e}")
    print(f"  - Intercept (R_base): {intercept:.2f} s")
    print("-" * 80)

    blind_actuals = [
        ("MiniCNN", 12, 139.67),
        ("MiniCNN", 20, 197.66),
        ("MiniCNN", 28, 243.48),
        ("TransformerBlock", 20, 249.50),
        ("TransformerBlock", 24, 325.89),
        ("TransformerBlock", 40, 460.37)
    ]

    print(f"{'Model Type':<18} | {'Size':<6} | {'Actual (s)':<12} | {'Predicted (s)':<15} | {'Error (%)'}")
    print("-" * 80)
    
    mapes = []
    for m_type, size, actual_time in blind_actuals:
        if m_type == "MiniCNN":
            model = MiniCNN(size).eval()
            dummy_input = torch.randn(1, 3, 16, 16)
        elif m_type == "TransformerBlock":
            model = TransformerBlockModel(size).eval()
            dummy_input = torch.randn(1, 16, size)
            
        feats = get_blind_features(model, dummy_input, m_type, size)
        if feats is None:
            continue
            
        pred_time = (feats["domain_fft_work"] * w_fft +
                     feats["domain_size"] * w_dom +
                     feats["num_rows"] * w_rows +
                     feats["total_assignments"] * w_assign +
                     feats["lookup_span"] * w_lookup +
                     feats["num_shuffles"] * w_shuf +
                     feats["total_const_size"] * w_const +
                     intercept)
        
        error = abs(actual_time - pred_time) / actual_time * 100
        mapes.append(error)
        
        print(f"{m_type:<18} | {size:<6} | {actual_time:<12.2f} | {pred_time:<15.2f} | {error:.2f}%")

    print("-" * 80)
    print(f"--> Overall Blind Test Mean Absolute Percentage Error (MAPE): {np.mean(mapes):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
