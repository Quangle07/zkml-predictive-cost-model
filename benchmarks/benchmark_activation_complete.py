import ezkl
import os
import time
import json
import torch
import torch.nn as nn
import asyncio
import numpy as np
import pandas as pd
import resource
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class ActivationModel(nn.Module):
    def __init__(self, activation_name):
        super(ActivationModel, self).__init__()
        if activation_name == "ReLU":
            self.activation = nn.ReLU()
        elif activation_name == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_name == "Tanh":
            self.activation = nn.Tanh()
        elif activation_name == "Linear":
            self.activation = nn.Identity()
            
    def forward(self, x):
        return self.activation(x)

async def run_benchmark():
    _to_test = ["linear", "ReLU", "Sigmoid", "Tanh"]
    tensor_sizes = [10, 100, 500] + list(range(1000, 20001, 500))
    num_trials = 5 
    
    base_path = os.getcwd()
    all_results = []

    # Grab the specific Eddie node we are running on
    current_node = socket.gethostname()
    
    print(" Starting Comprehensive ZK Benchmark (Time, RAM, Size)...")

    for act_name in activation_to_test:
        print(f"\n==================================================")
        print(f" ACTIVATION FUNCTION: {act_name}")
        print(f"==================================================")
        
        model = ActivationModel(act_name)
        model.eval()

        for size in tensor_sizes:
            print(f"\n--- esting Tensor Size: {size} ---")
            
            prefix = f"{act_name}_{size}"
            username = os.environ.get("USER")
            scratch_dir = "/exports/eddie/scratch/{username}"
            model_path = os.path.join(scratch_dir, f"{prefix}_model.onnx")
            settings_path = os.path.join(scratch_dir, f"{prefix}_settings.json")
            compiled_path = os.path.join(scratch_dir, f"{prefix}.compiled")
            pk_path = os.path.join(scratch_dir, f"{prefix}.pk")
            vk_path = os.path.join(scratch_dir, f"{prefix}.vk")
            srs_path = os.path.join(scratch_dir, f"{prefix}_kzg.srs")
            calib_path = os.path.join(scratch_dir, f"{prefix}_calib.json")
            
            dummy_x = torch.rand(1, size)
            
            torch.onnx.export(
                model, dummy_x, model_path, export_params=True, opset_version=18,
                input_names=['input'], output_names=['output']
            )

            with open(calib_path, "w") as f:
                json.dump({"input_data": [dummy_x.flatten().numpy().tolist()]}, f)

            ezkl.gen_settings(model_path, settings_path)
            ezkl.calibrate_settings(calib_path, model_path, settings_path, "resources")
            ezkl.compile_circuit(model_path, compiled_path, settings_path)
            
            await ezkl.get_srs(settings_path=settings_path, srs_path=srs_path)
            ezkl.setup(model=compiled_path, vk_path=vk_path, pk_path=pk_path, srs_path=srs_path)

            trial_proving_times = []
            trial_verify_times = []
            trial_proof_sizes = []

            for trial in range(num_trials):
                trial_data_path = os.path.join(scratch_dir, f"{prefix}_input_t{trial}.json")
                trial_witness_path = os.path.join(scratch_dir, f"{prefix}_witness_t{trial}.json")
                trial_proof_path = os.path.join(scratch_dir, f"{prefix}_proof_t{trial}.json")

                trial_x = torch.rand(1, size)
                with open(trial_data_path, "w") as f:
                    json.dump({"input_data": [trial_x.flatten().numpy().tolist()]}, f)

                ezkl.gen_witness(data=trial_data_path, model=compiled_path, output=trial_witness_path)

                # 1. Measure Proving Time
                start_time = time.time()
                ezkl.prove(
                    witness=trial_witness_path, model=compiled_path, pk_path=pk_path,
                    proof_path=trial_proof_path, srs_path=srs_path
                )
                trial_proving_times.append(time.time() - start_time)

                # 2. Measure Proof Size (KB)
                trial_proof_sizes.append(os.path.getsize(trial_proof_path) / 1024.0)

                # 3. Measure Verification Time
                start_verify = time.time()
                ezkl.verify(proof_path=trial_proof_path, settings_path=settings_path, vk_path=vk_path, srs_path=srs_path)
                trial_verify_times.append(time.time() - start_verify)

                os.remove(trial_data_path)
                os.remove(trial_witness_path)
                os.remove(trial_proof_path)

            # 4. Measure Peak Memory (RAM in MB)
            # Linux resource usage is in KB, so divide by 1024
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak_ram_mb = usage.ru_maxrss / 1024.0 

            mean_prove = np.mean(trial_proving_times)
            mean_verify = np.mean(trial_verify_times)
            mean_size = np.mean(trial_proof_sizes)
            
            print(f" {act_name} (Size {size}): Prove={mean_prove:.2f}s | Verify={mean_verify:.4f}s | RAM={peak_ram_mb:.0f}MB")
            
            all_results.append({
                "Activation": act_name,
                "Tensor_Size": size,
                "Mean_Proving_Time_s": mean_prove,
                "Std_Dev_Prove_s": np.std(trial_proving_times),
                "Mean_Verify_Time_s": mean_verify,
                "Mean_Proof_Size_KB": mean_size,
                "Peak_RAM_MB": peak_ram_mb
            })

            print(" Cleaning up heavy setup files...")
            for f_path in [model_path, model_path + ".data", settings_path, compiled_path, pk_path, vk_path, srs_path, calib_path]:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    
            pd.DataFrame(all_results).to_csv("activation_results_complete.csv", index=False)

    df = pd.DataFrame(all_results)
    colors = {"Linear": "grey", "ReLU": "blue", "Sigmoid": "red", "Tanh": "purple"}
    
    # Graph 1: Proving Time
    plt.figure(figsize=(10, 6))
    for act_name in activation_to_test:
        subset = df[df["Activation"] == act_name]
        plt.errorbar(subset["Tensor_Size"], subset["Mean_Proving_Time_s"], yerr=subset["Std_Dev_Prove_s"], fmt='-o', color=colors[act_name], label=act_name)
    plt.title("Proving Time vs Tensor Input Dimension")
    plt.xlabel("Tensor Elements")
    plt.ylabel("Seconds")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("graph_proving_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Graph 2: Peak RAM
    plt.figure(figsize=(10, 6))
    for act_name in activation_to_test:
        subset = df[df["Activation"] == act_name]
        plt.plot(subset["Tensor_Size"], subset["Peak_RAM_MB"], '-o', color=colors[act_name], label=act_name)
    plt.title("Peak Memory (RAM) vs Tensor Input Dimension")
    plt.xlabel("Tensor Elements")
    plt.ylabel("Megabytes (MB)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("graph_peak_ram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Graph 3: Verification Time
    plt.figure(figsize=(10, 6))
    for act_name in activation_to_test:
        subset = df[df["Activation"] == act_name]
        plt.plot(subset["Tensor_Size"], subset["Mean_Verify_Time_s"], '-o', color=colors[act_name], label=act_name)
    plt.title("Verification Time vs Tensor Input Dimension")
    plt.xlabel("Tensor Elements")
    plt.ylabel("Seconds")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("graph_verify_time.png", dpi=300, bbox_inches='tight')
    print("\n All graphs successfully saved!")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
