import json
import math
import os
import torch
import torch.nn as nn
import ezkl
import pandas as pd

# Define the Models
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

# Feature Extraction Routine
def extract_features(settings_path):
    with open(settings_path, "r") as file:
        settings = json.load(file)
    
    run_args = settings["run_args"]
    logrows = run_args["logrows"]
    domain_size = 1 << logrows
    
    lookup_min, lookup_max = run_args["lookup_range"]
    lookup_span = lookup_max - lookup_min
    
    features = {
        "logrows": logrows,
        "domain_size": domain_size,
        "domain_fft_work": domain_size * logrows,
        "num_rows": settings["num_rows"],
        "row_utilization": settings["num_rows"] / domain_size,
        "total_assignments": settings["total_assignments"],
        "num_inner_cols": run_args["num_inner_cols"],
        "total_const_size": settings["total_const_size"],
        "lookup_span": lookup_span,
        "lookup_log_size": math.ceil(math.log2(max(lookup_span, 1))),
        "required_lookup_types": len(settings["required_lookups"]),
        "required_range_checks": len(settings.get("required_range_checks", [])),
        "num_dynamic_lookups": settings.get("num_dynamic_lookups", 0),
        "total_dynamic_col_size": settings.get("total_dynamic_col_size", 0),
        "num_shuffles": settings.get("num_shuffles", 0),
        "total_shuffle_col_size": settings.get("total_shuffle_col_size", 0),
    }
    return features

# Extraction Engine
def process_model(model, dummy_input, model_name, size, actual_time):
    print(f"Extracting features for {model_name} (Size: {size})...")
    prefix = f"extract_{model_name}_{size}"
    onnx_path = f"{prefix}.onnx"
    data_path = f"data_{prefix}.json"
    settings_path = f"settings_{prefix}.json"

    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=14, do_constant_folding=True, dynamo=False)
    with open(data_path, "w") as f:
        json.dump(dict(input_data=[dummy_input.detach().numpy().reshape([-1]).tolist()]), f)

    try:
        ezkl.gen_settings(onnx_path, settings_path)
        ezkl.calibrate_settings(data_path, onnx_path, settings_path, "resources")
        
        features = extract_features(settings_path)
        features["model_type"] = model_name
        features["parameter_size"] = size
        features["actual_proving_time"] = actual_time
        return features
    except Exception as e:
        print(f"Failed on {model_name} {size}: {e}")
        return None
    finally:
        for f in [onnx_path, data_path, settings_path]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    extracted_data = []

    # Load actual median times from previous runs
    df_cnn = pd.read_csv("high_res_rigorous_results.csv")
    cnn_meds = df_cnn[df_cnn['model_type'] == 'MiniCNN'].groupby('parameter_size')['proving_time'].median().to_dict()

    df_trans = pd.read_csv("transformer_block_results.csv")
    trans_meds = df_trans.groupby('parameter_size')['proving_time'].median().to_dict()

    # Process CNNs
    for c, time in cnn_meds.items():
        model = MiniCNN(c).eval()
        dummy_input = torch.randn(1, 3, 16, 16)
        feat = process_model(model, dummy_input, "MiniCNN", c, time)
        if feat: extracted_data.append(feat)

    # Process Transformers
    for d, time in trans_meds.items():
        model = TransformerBlockModel(d).eval()
        dummy_input = torch.randn(1, 16, d)
        feat = process_model(model, dummy_input, "TransformerBlock", d, time)
        if feat: extracted_data.append(feat)

    # Save final dataset
    if extracted_data:
        pd.DataFrame(extracted_data).to_csv("circuit_features_master.csv", index=False)
        print("\nSuccess! Saved to circuit_features_master.csv")
