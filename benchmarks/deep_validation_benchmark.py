import time
import json
import os
import csv
import torch
import torch.nn as nn
import ezkl
import math
import pandas as pd

# Models
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
    def __init__(self, embed_dim, num_heads=2): # Reduced heads for awkward sizes
        super(TransformerBlockModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
    def forward(self, x):
        return self.encoder_layer(x)

# Expanded Feature Extraction
def extract_all_features(settings_path):
    with open(settings_path, "r") as file:
        settings = json.load(file)
    run_args = settings["run_args"]
    logrows = run_args["logrows"]
    domain_size = 1 << logrows
    lookup_min, lookup_max = run_args["lookup_range"]

    return {
        "logrows": logrows,
        "domain_size": domain_size,
        "domain_fft_work": domain_size * logrows,
        "num_rows": settings["num_rows"],
        "row_utilization": settings["num_rows"] / domain_size,
        "total_assignments": settings["total_assignments"],
        "num_inner_cols": run_args["num_inner_cols"],
        "total_const_size": settings["total_const_size"],
        "lookup_span": lookup_max - lookup_min,
        "num_shuffles": settings.get("num_shuffles", 0),
        "total_shuffle_col_size": settings.get("total_shuffle_col_size", 0)
    }

# Execution Pipeline
def run_benchmark(model, dummy_input, model_name, size_param):
    print(f"\n--- Deep Validation: {model_name} | Size: {size_param} ---")
    prefix = f"deep_val_{model_name}_{size_param}"
    onnx_path = f"{prefix}.onnx"
    data_path = f"data_{prefix}.json"
    settings_path = f"settings_{prefix}.json"
    compiled_path = f"network_{prefix}.compiled"
    pk_path = f"pk_{prefix}.key"
    vk_path = f"vk_{prefix}.key"
    witness_path = f"witness_{prefix}.json"
    proof_path = f"proof_{prefix}.json"

    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=14, do_constant_folding=True, dynamo=False)
    with open(data_path, "w") as f:
        json.dump(dict(input_data=[dummy_input.detach().numpy().reshape([-1]).tolist()]), f)

    try:
        ezkl.gen_settings(onnx_path, settings_path)
        ezkl.calibrate_settings(data_path, onnx_path, settings_path, "resources")

        # Extract features BEFORE compiling so we have them even if it crashes
        features = extract_all_features(settings_path)

        ezkl.compile_circuit(onnx_path, compiled_path, settings_path)
        ezkl.setup(compiled_path, vk_path, pk_path)
        ezkl.gen_witness(data_path, compiled_path, witness_path)

        start_time = time.time()
        ezkl.prove(witness_path, compiled_path, pk_path, proof_path)
        proving_time = time.time() - start_time

        print(f"Success -> Time: {proving_time:.2f}s")

        features["model_type"] = model_name
        features["parameter_size"] = size_param
        features["actual_proving_time"] = proving_time

        return features

    except Exception as e:
        print(f"Failed: {e}")
        return None
    finally:
        for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
        csv_file = "deep_validation_results.csv"

        # New Unseen CNN Sizes
        for c in [14, 22, 30]:
            feat = run_benchmark(MiniCNN(c).eval(), torch.randn(1, 3, 16, 16), "MiniCNN", c)
            if feat:
                df = pd.DataFrame([feat])
                df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

        # New Unseen Transformer Sizes
        for d in [18, 26, 34]:
            feat = run_benchmark(TransformerBlockModel(d, num_heads=2).eval(), torch.randn(1, 16, d), "TransformerBlock", d)
            if feat:
                df = pd.DataFrame([feat])
                df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

        print("\nAll models finished successfully!")
