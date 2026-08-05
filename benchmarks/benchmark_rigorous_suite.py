import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os
import numpy as np

# Define All Models
class LinearModel(nn.Module):
    def __init__(self, size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(1, size)
    def forward(self, x):
        return self.fc(x)

class SigmoidModel(nn.Module):
    def __init__(self):
        super(SigmoidModel, self).__init__()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(x)

class Conv2DModel(nn.Module):
    def __init__(self, out_channels):
        super(Conv2DModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)

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

# Benchmarking Engine
def run_benchmark(model, dummy_input, model_name, size_param, trial):
    print(f"\n--- Testing {model_name} | Size/Channels: {size_param} | Trial: {trial} ---")

    prefix = f"{model_name}_{size_param}_{trial}"
    onnx_path = f"{prefix}.onnx"
    data_path = f"data_{prefix}.json"
    settings_path = f"settings_{prefix}.json"
    compiled_path = f"network_{prefix}.compiled"
    pk_path = f"pk_{prefix}.key"
    vk_path = f"vk_{prefix}.key"
    witness_path = f"witness_{prefix}.json"
    proof_path = f"proof_{prefix}.json"

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14,
        do_constant_folding=True, dynamo=False
    )

    with open(data_path, "w") as f:
        json.dump(dict(input_data=[dummy_input.detach().numpy().reshape([-1]).tolist()]), f)

    try:
        ezkl.gen_settings(onnx_path, settings_path)
        with open(settings_path, "r") as f:
            logrows = json.load(f)["run_args"]["logrows"]

        ezkl.compile_circuit(onnx_path, compiled_path, settings_path)
        ezkl.setup(compiled_path, vk_path, pk_path)
        ezkl.gen_witness(data_path, compiled_path, witness_path)

        tracemalloc.start()
        start_time = time.time()
        ezkl.prove(witness_path, compiled_path, pk_path, proof_path)
        proving_time = time.time() - start_time
        
        _, peak_ram_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ram_mb = peak_ram_bytes / (1024 * 1024)
        proof_size_mb = os.path.getsize(proof_path) / (1024 * 1024)

        print(f"Result -> Time: {proving_time:.2f}s | RAM: {peak_ram_mb:.2f}MB | logrows: {logrows}")

        return {
            "model_type": model_name,
            "parameter_size": size_param,
            "trial": trial,
            "logrows": logrows,
            "proving_time": proving_time,
            "peak_ram_mb": peak_ram_mb,
            "proof_size_mb": proof_size_mb
        }

    except Exception as e:
        print(f"Failed: {e}")
        return None

    finally:
        for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
            if os.path.exists(f):
                os.remove(f)

# Main Execution Block
if __name__ == "__main__":
    TRIALS = 5
    all_results = []

    # Parameter ranges
    linear_sizes = list(range(500, 20001, 500))
    sigmoid_sizes = list(range(500, 20001, 500))
    conv_channels = list(range(4, 65, 4))
    cnn_channels = list(range(8, 65, 8)) # Step by 8 up to 64

    # Run Linear
    for size in linear_sizes:
        for t in range(1, TRIALS + 1):
            model = LinearModel(size).eval()
            res = run_benchmark(model, torch.randn(1, 1), "Linear", size, t)
            if res: all_results.append(res)

    # Run Sigmoid
    for size in sigmoid_sizes:
        for t in range(1, TRIALS + 1):
            model = SigmoidModel().eval()
            res = run_benchmark(model, torch.randn(1, size), "Sigmoid", size, t)
            if res: all_results.append(res)

    # Run Conv2D
    for channels in conv_channels:
        for t in range(1, TRIALS + 1):
            model = Conv2DModel(channels).eval()
            res = run_benchmark(model, torch.randn(1, 3, 16, 16), "Conv2D", channels, t)
            if res: all_results.append(res)

    # Run Composite CNN
    for channels in cnn_channels:
        for t in range(1, TRIALS + 1):
            model = MiniCNN(channels).eval()
            res = run_benchmark(model, torch.randn(1, 3, 16, 16), "MiniCNN", channels, t)
            if res: all_results.append(res)

    # Save flat CSV
    if all_results:
        with open("high_res_rigorous_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print("\n Benchmark Complete! Data saved to high_res_rigorous_results.csv")
