import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os

# Define the Full Transformer Block
class TransformerBlockModel(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(TransformerBlockModel, self).__init__()
        # A standard LLM building block: Attention + Add + LayerNorm + Linear + GELU
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu",
            batch_first=True
        )
        
    def forward(self, x):
        return self.encoder_layer(x)

# Benchmarking Engine
def run_benchmark(model, dummy_input, model_name, size_param, trial):
    print(f"\n--- Testing {model_name} | Dim: {size_param} | Trial: {trial} ---")

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

        # INCREMENTAL SAVE 
        with open("transformer_block_results.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model_type", "parameter_size", "trial", "logrows", "proving_time", "peak_ram_mb", "proof_size_mb"])
            writer.writerow({
                "model_type": model_name,
                "parameter_size": size_param,
                "trial": trial,
                "logrows": logrows,
                "proving_time": proving_time,
                "peak_ram_mb": peak_ram_mb,
                "proof_size_mb": proof_size_mb
            })

    except Exception as e:
        print(f"Failed: {e}")

    finally:
        for f in [onnx_path, data_path, settings_path, compiled_path, pk_path, vk_path, witness_path, proof_path]:
            if os.path.exists(f):
                os.remove(f)

# Main Execution Block
if __name__ == "__main__":
    # Initialise CSV
    if not os.path.exists("transformer_block_results.csv"):
        with open("transformer_block_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model_type", "parameter_size", "trial", "logrows", "proving_time", "peak_ram_mb", "proof_size_mb"])
            writer.writeheader()

    TRIALS = 3
    # Keeping dims slightly smaller than isolated attention because the full block is massive
    block_dims = [16, 32, 64, 128] 
    seq_len = 16

    for dim in block_dims:
        for t in range(1, TRIALS + 1):
            model = TransformerBlockModel(embed_dim=dim, num_heads=4).eval()
            run_benchmark(model, torch.randn(1, seq_len, dim), "TransformerBlock", dim, t)

    print("\n Full Transformer Block Benchmark Complete!")
