import time
import tracemalloc
import json
import csv
import torch
import torch.nn as nn
import torch.onnx
import ezkl
import os

# Define Modern ML Models
class GELUModel(nn.Module):
    def __init__(self):
        super(GELUModel, self).__init__()
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.gelu(x)

class LayerNormModel(nn.Module):
    def __init__(self, embed_dim):
        super(LayerNormModel, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        return self.norm(x)

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttentionModel, self).__init__()
        # batch_first=True makes the input shape (batch, seq_len, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        # Self-attention: query, key, and value are all the same input
        attn_output, _ = self.mha(x, x, x)
        return attn_output

# Benchmarking Engine
def run_benchmark(model, dummy_input, model_name, size_param, trial):
    print(f"\n--- Testing {model_name} | Size/Dim: {size_param} | Trial: {trial} ---")

    prefix = f"{model_name}_{size_param}_{trial}"
    onnx_path = f"{prefix}.onnx"
    data_path = f"data_{prefix}.json"
    settings_path = f"settings_{prefix}.json"
    compiled_path = f"network_{prefix}.compiled"
    pk_path = f"pk_{prefix}.key"
    vk_path = f"vk_{prefix}.key"
    witness_path = f"witness_{prefix}.json"
    proof_path = f"proof_{prefix}.json"

    # Exporting Attention requires setting opset_version to at least 14
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
    TRIALS = 3  # Reduced to 3 because Transformers take significantly longer to prove
    all_results = []
    csv_filename = "transformer_benchmark_results.csv"

    # Parameter sweeps
    gelu_sizes = list(range(500, 10001, 500))
    layernorm_dims = [64, 128, 256, 512, 1024]
    
    # Attention is massive. We keep sequence length fixed at 16, and scale the embedding dimension.
    attention_dims = [16, 32, 64, 128, 256] 
    seq_len = 16

    # Run GELU
    for size in gelu_sizes:
        for t in range(1, TRIALS + 1):
            model = GELUModel().eval()
            res = run_benchmark(model, torch.randn(1, size), "GELU", size, t)
            if res: all_results.append(res)

    # Run LayerNorm
    for dim in layernorm_dims:
        for t in range(1, TRIALS + 1):
            model = LayerNormModel(dim).eval()
            # Input shape: (batch_size, seq_len, embed_dim)
            res = run_benchmark(model, torch.randn(1, seq_len, dim), "LayerNorm", dim, t)
            if res: all_results.append(res)

    # Run Self-Attention
    for dim in attention_dims:
        for t in range(1, TRIALS + 1):
            model = SelfAttentionModel(embed_dim=dim, num_heads=4).eval()
            res = run_benchmark(model, torch.randn(1, seq_len, dim), "SelfAttention", dim, t)
            if res: all_results.append(res)

    # Save to CSV
    if all_results:
        with open(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print("\n Transformer Benchmark Complete! Data saved to transformer_benchmark_results.csv")
