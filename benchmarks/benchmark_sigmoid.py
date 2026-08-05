import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os

# 1. Define Model (Sigmoid Activation)
class SigmoidModel(nn.Module):
    def __init__(self):
        super(SigmoidModel, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)

tensor_sizes = list(range(500, 10001, 500))

results = []

for size in tensor_sizes:
    print(f"\n--- Testing Sigmoid Tensor Size: {size} ---")

    # Paths
    onnx_path = f"sigmoid_{size}.onnx"
    data_path = f"data_{size}.json"
    settings_path = f"settings_{size}.json"
    compiled_path = f"network_{size}.compiled"
    pk_path = f"pk_{size}.key"
    vk_path = f"vk_{size}.key"
    witness_path = f"witness_{size}.json"
    proof_path = f"proof_{size}.json"

    # Export model & dummy input tensor of shape (1, size)
    model = SigmoidModel().eval()
    dummy_input = torch.randn(1, size)

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

        print(f"Size: {size} | logrows: {logrows} | Time: {proving_time:.2f}s | Peak RAM: {peak_ram_mb:.2f}MB | Proof Size: {proof_size_mb:.2f}MB")

        results.append({
            "size": size,
            "logrows": logrows,
            "proving_time": proving_time,
            "peak_ram_mb": peak_ram_mb,
            "proof_size_mb": proof_size_mb
        })

    except Exception as e:
        print(f"Failed at size {size}: {e}")

    finally:
        for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
            if os.path.exists(f):
                os.remove(f)

# Save results
with open("sigmoid_results.json", "w") as f:
    json.dump(results, f, indent=4)

if results:
    keys = results[0].keys()
    with open("sigmoid_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

print("\nBenchmark complete! Results saved to sigmoid_results.json and sigmoid_results.csv")
