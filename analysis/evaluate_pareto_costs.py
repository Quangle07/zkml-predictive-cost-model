import json
import glob
import math

# Coefficients from the cost model
W_FFT = 2.9350e-8
W_ASSIGN = 1.5115e-4
W_LOOKUP = 9.7656e-4
W_CONST = 6.6258e-2

settings_files = glob.glob("onnx_models/settings_*.json")

print(f"{'Configuration':<25} | {'Logrows':<8} | {'Est. Time (s)':<15}")
print("-" * 55)

for path in sorted(settings_files):
    with open(path, 'r') as f:
        s = json.load(f)

    run_args = s.get("run_args", {})
    k = run_args.get("logrows", 17)
    bits = run_args.get("bits", 16)
    d_size = 2 ** k

    total_assignments = s.get("total_assignments", 0)
    const_size = s.get("total_const_size", 0)

    # EZKL scales non-linear lookup tables strictly by 2^bits.
    # Calculate Non-Linear Lookup Span (L_span)
    if "sigmoid" in path or "gelu" in path:
        lookup_span = 2 ** bits
    else:
        lookup_span = 0

    # Apply cost formula
    fft_work = d_size * math.log2(d_size)
    t_prove = (W_FFT * fft_work) + (W_ASSIGN * total_assignments) + (W_LOOKUP * lookup_span) + (W_CONST * const_size)

    config_name = path.split('/')[-1].replace('settings_', '').replace('.json', '')
    print(f"{config_name:<25} | {k:<8} | {t_prove:<15.2f}")
