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
    activations_to_test = ["Linear", "ReLU", "Sigmoid", "Tanh"]
    scales_to_test = [13, 14, 15, 16]
    fixed_size = 2000
    num_trials = 3
    username = os.environ.get("USER") 
    scratch_dir = f"/exports/eddie/scratch/{username}"
    base_path = os.getcwd()
    all_results = []

    print(f" Starting quantisation Benchmark (Fixed Size: {fixed_size})...")

    for act_name in activations_to_test:
        print(f"\n==================================================")
        print(f" ACTIVATION FUNCTION: {act_name}")
        print(f"==================================================")

        model = ActivationModel(act_name)
        model.eval()

        for scale in scales_to_test:
            print(f"\n--- esting Scale: {scale}-bit ---")

            prefix = f"{act_name}_scale{scale}"
            model_path = os.path.join(scratch_dir, f"{prefix}_model.onnx")
            settings_path = os.path.join(scratch_dir, f"{prefix}_settings.json")
            compiled_path = os.path.join(scratch_dir, f"{prefix}.compiled")
            pk_path = os.path.join(scratch_dir, f"{prefix}.pk")
            vk_path = os.path.join(scratch_dir, f"{prefix}.vk")
            srs_path = os.path.join(scratch_dir, f"{prefix}_kzg.srs")
            calib_path = os.path.join(scratch_dir, f"{prefix}_calib.json")

            dummy_x = torch.rand(1, fixed_size)

            try:
                # Export ONNX
                torch.onnx.export(
                    model, dummy_x, model_path, export_params=True, opset_version=18,
                    input_names=['input'], output_names=['output']
                )

                # Generate defaults and calibrate first (sets up logs/structures)
                ezkl.gen_settings(model_path, settings_path)

                with open(calib_path, "w") as f:
                    json.dump({"input_data": [dummy_x.flatten().numpy().tolist()]}, f)

                ezkl.calibrate_settings(calib_path, model_path, settings_path, "resources")

                # Intercept the file and force our scale over the calibrated ones
                with open(settings_path, "r") as f:
                    settings = json.load(f)

                settings["run_args"]["input_scale"] = scale
                settings["run_args"]["param_scale"] = scale
                settings["run_args"]["lookup_bits"] = max(scale + 1, 12) 

                # Force the circuit size to be large enough for the lookup table
                settings["run_args"]["logrows"] = max(settings["run_args"]["logrows"], scale + 2)

                with open(settings_path, "w") as f:
                    json.dump(settings, f)

                # Compile with our forced settings
                ezkl.compile_circuit(model_path, compiled_path, settings_path)

                # Set up ZK 
                await ezkl.get_srs(settings_path=settings_path, srs_path=srs_path)
                ezkl.setup(model=compiled_path, vk_path=vk_path, pk_path=pk_path, srs_path=srs_path)

                trial_proving_times = []
                trial_verify_times = []
                trial_proof_sizes = []

                for trial in range(num_trials):
                    trial_data_path = os.path.join(scratch_dir, f"{prefix}_input_t{trial}.json")
                    trial_witness_path = os.path.join(scratch_dir, f"{prefix}_witness_t{trial}.json")
                    trial_proof_path = os.path.join(scratch_dir, f"{prefix}_proof_t{trial}.json")

                    trial_x = torch.rand(1, fixed_size)
                    with open(trial_data_path, "w") as f:
                        json.dump({"input_data": [trial_x.flatten().numpy().tolist()]}, f)

                    ezkl.gen_witness(data=trial_data_path, model=compiled_path, output=trial_witness_path)

                    start_time = time.time()
                    ezkl.prove(
                        witness=trial_witness_path, model=compiled_path, pk_path=pk_path,
                        proof_path=trial_proof_path, srs_path=srs_path
                    )
                    trial_proving_times.append(time.time() - start_time)

                    trial_proof_sizes.append(os.path.getsize(trial_proof_path) / 1024.0)

                    start_verify = time.time()
                    ezkl.verify(proof_path=trial_proof_path, settings_path=settings_path, vk_path=vk_path, srs_path=srs_path)
                    trial_verify_times.append(time.time() - start_verify)

                    os.remove(trial_data_path)
                    os.remove(trial_witness_path)
                    os.remove(trial_proof_path)

                # Get Memory Usage
                usage = resource.getrusage(resource.RUSAGE_SELF)
                peak_ram_mb = usage.ru_maxrss / 1024.0

                mean_prove = np.mean(trial_proving_times)
                
                print(f" {act_name} (Scale {scale}): Prove={mean_prove:.2f}s | RAM={peak_ram_mb:.0f}MB")

                all_results.append({
                    "Activation": act_name,
                    "Scale": scale,
                    "Mean_Proving_Time_s": mean_prove,
                    "Mean_Proof_Size_KB": np.mean(trial_proof_sizes),
                    "Peak_RAM_MB": peak_ram_mb,
                    "Status": "Success"
                })

            except Exception as e:
                print(f"X CRASH on {act_name} at Scale {scale}: {str(e)}")
                all_results.append({
                    "Activation": act_name,
                    "Scale": scale,
                    "Mean_Proving_Time_s": None,
                    "Mean_Proof_Size_KB": None,
                    "Peak_RAM_MB": None,
                    "Status": "Failed/OOM"
                })

            finally:
                print(" Cleaning up setup files...")
                for f_path in [model_path, model_path + ".data", settings_path, compiled_path, pk_path, vk_path, srs_path, calib_path]:
                    if os.path.exists(f_path):
                        os.remove(f_path)
                
                # Checkpoint data
                pd.DataFrame(all_results).to_csv("all_quantisation_results.csv", index=False)

    # GRAPHING
    df = pd.DataFrame(all_results)
    df_success = df[df["Status"] == "Success"]
    colors = {"Linear": "grey", "ReLU": "blue", "Sigmoid": "red", "Tanh": "purple"}

    # Graph 1: Peak RAM vs Scale
    plt.figure(figsize=(10, 6))
    for act_name in activations_to_test:
        subset = df_success[df_success["Activation"] == act_name]
        plt.plot(subset["Scale"], subset["Peak_RAM_MB"], '-o', color=colors[act_name], label=act_name)
    plt.title(f"Quantisation Penalty: Peak RAM vs Bit-Scale (Size={fixed_size})")
    plt.xlabel("Quantisation Scale (Bits)")
    plt.ylabel("Megabytes (MB)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("graph_quantisation_ram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Graph 2: Proving Time vs Scale
    plt.figure(figsize=(10, 6))
    for act_name in activations_to_test:
        subset = df_success[df_success["Activation"] == act_name]
        plt.plot(subset["Scale"], subset["Mean_Proving_Time_s"], '-o', color=colors[act_name], label=act_name)
    plt.title(f"Quantisation Penalty: Proving Time vs Bit-Scale (Size={fixed_size})")
    plt.xlabel("Quantisation Scale (Bits)")
    plt.ylabel("Seconds")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("graph_quantisation_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n Quantisation benchmark complete. Graphs saved!")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
