import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os

# Define the models
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
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu",
            batch_first=True
        )
    def forward(self, x):
        return self.encoder_layer(x)

def run_benchmark(model, dummy_input, model_name, size_param, trial):
    print(f"\n--- Blind Test {model_name} | Size: {size_param} | Trial: {trial} ---")
    prefix = f"blind_{model_name}_{size_param}_{trial}"
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
        ezkl.compile_circuit(onnx_path, compiled_path, settings_path)
        ezkl.setup(compiled_path, vk_path, pk_path)
        ezkl.gen_witness(data_path, compiled_path, witness_path)

        start_time = time.time()
        ezkl.prove(witness_path, compiled_path, pk_path, proof_path)
        proving_time = time.time() - start_time

        print(f"Result -> Time: {proving_time:.2f}s")
        
        with open("blind_test_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_name, size_param, trial, proving_time])

    except Exception as e:
        print(f"Failed: {e}")

    finally:
        for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    if not os.path.exists("blind_test_results.csv"):
        with open("blind_test_results.csv", "w", newline="") as f:
            csv.writer(f).writerow(["model_type", "parameter_size", "trial", "proving_time"])

    # CNN Validations
    for c in [12, 20, 28]:
        for t in range(1, 4):
            run_benchmark(MiniCNN(c).eval(), torch.randn(1, 3, 16, 16), "MiniCNN", c, t)

    # Transformer Validations (Keep dim 40 to avoid 64GB OOM limit)
    for d in [20, 24, 40]:
        for t in range(1, 4):
            run_benchmark(TransformerBlockModel(d, num_heads=4).eval(), torch.randn(1, 16, d), "TransformerBlock", d, t)
