import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os

class LinearModel(nn.Module):
    def __init__(self, size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(1, size)
    def forward(self, x):
        return self.fc(x)

sizes = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
trials = 3
csv_filename = "logrows_sweep_results.csv"

# Initialise CSV with headers if it doesn't exist
fieldnames = ["size", "trial", "logrows", "proving_time", "peak_ram_mb", "proof_size_mb"]
if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

for size in sizes:
    for trial in range(1, trials + 1):
        print(f"\n--- Testing Linear Size: {size} | Trial: {trial}/{trials} ---")

        prefix = f"sweep_{size}_{trial}"
        onnx_path = f"{prefix}.onnx"
        data_path = f"data_{prefix}.json"
        settings_path = f"settings_{prefix}.json"
        compiled_path = f"network_{prefix}.compiled"
        pk_path = f"pk_{prefix}.key"
        vk_path = f"vk_{prefix}.key"
        witness_path = f"witness_{prefix}.json"
        proof_path = f"proof_{prefix}.json"

        model = LinearModel(size).eval()
        dummy_input = torch.randn(1, 1)

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

            print(f"Size: {size} | logrows: {logrows} | Time: {proving_time:.2f}s | RAM: {peak_ram_mb:.2f}MB")

            # INCREMENTAL SAVE: Write to CSV immediately
            with open(csv_filename, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    "size": size,
                    "trial": trial,
                    "logrows": logrows,
                    "proving_time": proving_time,
                    "peak_ram_mb": peak_ram_mb,
                    "proof_size_mb": proof_size_mb
                })

        except Exception as e:
            print(f"Failed at size {size}, trial {trial}: {e}")

        finally:
            for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
                if os.path.exists(f):
                    os.remove(f)

print("\nSweep complete! Results saved to logrows_sweep_results.csv")
