import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os

# Define CNN
class MiniCNN(nn.Module):
    def __init__(self, out_channels):
        super(MiniCNN, self).__init__()
        # Conv2d -> ReLU -> MaxPool2d -> Linear
        self.conv = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces 16x16 -> 8x8
        self.fc = nn.Linear(out_channels * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Sweep output channels from 8 up to 96
channel_sizes = list(range(8, 97, 8))
results = []

for channels in channel_sizes:
    print(f"\n--- Testing Mini-CNN Channels: {channels} ---")

    # Paths
    onnx_path = f"cnn_{channels}.onnx"
    data_path = f"data_{channels}.json"
    settings_path = f"settings_{channels}.json"
    compiled_path = f"network_{channels}.compiled"
    pk_path = f"pk_{channels}.key"
    vk_path = f"vk_{channels}.key"
    witness_path = f"witness_{channels}.json"
    proof_path = f"proof_{channels}.json"

    # Input: Standard RGB 16x16 image tensor
    model = MiniCNN(channels).eval()
    dummy_input = torch.randn(1, 3, 16, 16)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14,
        do_constant_folding=True, dynamo=False
    )

    with open(data_path, "w") as f:
        json.dump(dict(input_data=[dummy_input.detach().numpy().reshape([-1]).tolist()]), f)

    try:
        ezkl.gen_settings(onnx_path, settings_path)
        
        # Extract logrows for the predictive cost model
        with open(settings_path, "r") as f:
            settings_data = json.load(f)
            logrows = settings_data["run_args"]["logrows"]

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

        print(f"Channels: {channels} | logrows: {logrows} | Time: {proving_time:.2f}s | Peak RAM: {peak_ram_mb:.2f}MB | Proof Size: {proof_size_mb:.2f}MB")

        results.append({
            "channels": channels,
            "logrows": logrows,
            "proving_time": proving_time,
            "peak_ram_mb": peak_ram_mb,
            "proof_size_mb": proof_size_mb
        })

    except Exception as e:
        print(f"Failed at channels {channels}: {e}")

    finally:
        for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
            if os.path.exists(f):
                os.remove(f)

# Save final benchmark results
with open("cnn_results.json", "w") as f:
    json.dump(results, f, indent=4)

if results:
    keys = results[0].keys()
    with open("cnn_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

print("\nBenchmark complete! Results saved to cnn_results.json and cnn_results.csv")
