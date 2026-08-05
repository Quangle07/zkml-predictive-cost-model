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
import socket

class OperatorModel(nn.Module):
    def __init__(self, op_name):
        super(OperatorModel, self).__init__()
        # Standard 3 channel input (like an RGB image)
        if op_name == "Conv2D":
            self.op = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        elif op_name == "MaxPool2D":
            self.op = nn.MaxPool2d(kernel_size=2)
        elif op_name == "AvgPool2D":
            self.op = nn.AvgPool2d(kernel_size=2)
        elif op_name == "BatchNorm2D":
            self.op = nn.BatchNorm2d(num_features=3)

    def forward(self, x):
        return self.op(x)

async def run_benchmark():
    operators = ["Conv2D", "MaxPool2D", "AvgPool2D", "BatchNorm2D"]
    # Image resolutions: 8x8, 16x16, 32x32, 64x64
    resolutions = [8, 16, 32, 64]
    num_trials = 3
    
    username = os.environ.get("USER")
    scratch_dir = "/exports/eddie/scratch/{username}"
    all_results = []
    
    # What Eddie node we are running on 
    current_node = socket.gethostname()

    print(f" Starting Operator Benchmark on node: {current_node}")

    for op_name in operators:
        print(f"\n==================================================")
        print(f" OPERATOR: {op_name}")
        print(f"==================================================")

        model = OperatorModel(op_name)
        model.eval()

        for res in resolutions:
            total_elements = 3 * res * res
            print(f"\n--- Testing Resolution: {res}x{res} (Total Elements: {total_elements}) ---")

            prefix = f"{op_name}_res{res}"
            model_path = os.path.join(scratch_dir, f"{prefix}_model.onnx")
            settings_path = os.path.join(scratch_dir, f"{prefix}_settings.json")
            compiled_path = os.path.join(scratch_dir, f"{prefix}.compiled")
            pk_path = os.path.join(scratch_dir, f"{prefix}.pk")
            vk_path = os.path.join(scratch_dir, f"{prefix}.vk")
            srs_path = os.path.join(scratch_dir, f"{prefix}_kzg.srs")
            calib_path = os.path.join(scratch_dir, f"{prefix}_calib.json")

            # 4D Tensor: (Batch=1, Channels=3, Height=res, Width=res)
            dummy_x = torch.rand(1, 3, res, res)

            try:
                torch.onnx.export(
                    model, dummy_x, model_path, export_params=True, opset_version=18,
                    input_names=['input'], output_names=['output']
                )

                ezkl.gen_settings(model_path, settings_path)

                with open(calib_path, "w") as f:
                    json.dump({"input_data": [dummy_x.flatten().numpy().tolist()]}, f)

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

                    trial_x = torch.rand(1, 3, res, res)
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

                usage = resource.getrusage(resource.RUSAGE_SELF)
                peak_ram_mb = usage.ru_maxrss / 1024.0

                print(f" {op_name} ({res}x{res}): Prove={np.mean(trial_proving_times):.2f}s | RAM={peak_ram_mb:.0f}MB")

                all_results.append({
                    "Operator": op_name,
                    "Resolution": res,
                    "Total_Elements": total_elements,
                    "Mean_Proving_Time_s": np.mean(trial_proving_times),
                    "Mean_Verify_Time_s": np.mean(trial_verify_times),
                    "Mean_Proof_Size_KB": np.mean(trial_proof_sizes),
                    "Peak_RAM_MB": peak_ram_mb,
                    "Compute_Node": current_node,
                    "Status": "Success"
                })

            except Exception as e:
                print(f" CRASH on {op_name} at {res}x{res}: {str(e)}")
                all_results.append({
                    "Operator": op_name,
                    "Resolution": res,
                    "Total_Elements": total_elements,
                    "Status": f"Failed: {str(e)[:50]}"
                })

            finally:
                for f_path in [model_path, model_path + ".data", settings_path, compiled_path, pk_path, vk_path, srs_path, calib_path]:
                    if os.path.exists(f_path):
                        os.remove(f_path)
                
                pd.DataFrame(all_results).to_csv("operator_benchmark_results.csv", index=False)

    print("\n Operator benchmark complete. Data saved to operator_benchmark_results.csv!")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
