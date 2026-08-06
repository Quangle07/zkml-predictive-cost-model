import ezkl
import os
import json
import torch
import onnx

models = [
    "model_a_mlp", "model_b_cnn", "model_c_sigmoid",
    "model_d_tinycnn", "model_e_deepmlp", "model_f_gelu",
    "model_g_mediumcnn", "model_h_ultralightmlp"
]

# Sweep 5 different quantisation scales
bit_widths = [16, 12, 8, 6, 4]
scales = {16: 7, 12: 5, 8: 3, 6: 2, 4: 1}

print("--- Starting EZKL Compilation & Quantisation Sweep ---")

for model_name in models:
    raw_onnx_path = f"onnx_models/{model_name}.onnx"
    cleaned_onnx_path = f"onnx_models/{model_name}_cleaned.onnx"

    # Clean and re-save the ONNX model to make it tract-friendly
    try:
        model_proto = onnx.load(raw_onnx_path)
        onnx.save(model_proto, cleaned_onnx_path)
    except Exception as e:
        print(f"Failed to load/clean ONNX for {model_name}: {e}")
        continue

    for bits in bit_widths:
        print(f"\nProcessing {model_name} at {bits}-bit...")
        settings_path = f"onnx_models/settings_{model_name}_{bits}bit.json"
        compiled_path = f"onnx_models/{model_name}_{bits}bit.compiled"

        try:
            # Generate settings using the cleaned ONNX file
            ezkl.gen_settings(cleaned_onnx_path, settings_path)

            # Open settings and apply quantization args
            with open(settings_path, "r") as f:
                settings = json.load(f)

            settings["run_args"]["bits"] = bits
            settings["run_args"]["scale"] = scales[bits]

            with open(settings_path, "w") as f:
                json.dump(settings, f)

            # Compile the circuit
            ezkl.compile_circuit(cleaned_onnx_path, compiled_path, settings_path)
            print(f"Success! Compiled {compiled_path}")

        except Exception as e:
            print(f"Failed to compile {model_name} at {bits}-bit. Error: {e}")

print("\n--- Compilation Sweep Complete ---")
