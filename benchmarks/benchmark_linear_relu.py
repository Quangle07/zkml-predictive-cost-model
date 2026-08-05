import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os

# 1. Define Fused Model (Linear + ReLU + Linear sandwich)
class FusedModel(nn.Module):
    def __init__(self, size):
        super(FusedModel, self).__init__()
        self.linear1 = nn.Linear(1, size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(size, 1)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Range of tensor sizes to test
tensor_sizes = list(range(500, 20001, 500))

results = []

for size in tensor_sizes:
    print(f"\n--- Testing Tensor Size: {size} ---")

    # Paths
    onnx_path = f"fused_{size}.onnx"
    data_path = f"data_{size}.json"
    settings_path = f"settings_{size}.json"
    compiled_path = f"network_{size}.compiled"
    pk_path = f"pk_{size}.key"
    vk_path = f"vk_{size}.key"
    witness_path = f"witness_{size}.json"
    proof_path = f"proof_{size}.json"

    # Export model & dummy data
    model = FusedModel(size).eval()
    dummy_input = torch.randn(1, 1)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14,
        do_constant_folding=True, dynamo=False
    )

    with open(data_path, "w") as f:
        json.dump(dict(input_data=[dummy_input.detach().numpy().reshape([-1]).tolist()]), f)

    try:
        # Settings & Compilation
        ezkl.gen_settings(onnx_path, settings_path)
        ezkl.compile_circuit(onnx_path, compiled_path, settings_path)
        ezkl.setup(compiled_path, vk_path, pk_path)

        ezkl.gen_witness(data_path, compiled_path, witness_path)

        # Measure Memory & Proving Time
        tracemalloc.start()
        start_time = time.time()

        # Pass the WITNESS path, not the data path
        ezkl.prove(witness_path, compiled_path, pk_path, proof_path)

        proving_time = time.time() - start_time
        _, peak_ram_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ram_mb = peak_ram_bytes / (1024 * 1024)
        proof_size_mb = os.path.getsize(proof_path) / (1024 * 1024)

        print(f"Size: {size} | Time: {proving_time:.2f}s | Peak RAM: {peak_ram_mb:.2f}MB | Proof Size: {proof_size_mb:.2f}MB")

        results.append({
            "size": size,
            "proving_time": proving_time,
            "peak_ram_mb": peak_ram_mb,
            "proof_size_mb": proof_size_mb
        })

    except Exception as e:
        print(f"Failed at size {size}: {e}")

    finally:
        # Cleanup temporary build files
        for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
            if os.path.exists(f):
                os.remove(f)

# Save final benchmark results
with open("combinatorial_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Save final benchmark results as CSV
if results:
    keys = results[0].keys()
    with open("combinatorial_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

print("\nBenchmark complete! Results saved to combinatorial_results.json and combinatorial_results.csv")
