## **Predictive Cost Modelling and Empirical Characterisation of ZKML Circuits:**

An empirical benchmark suite and predictive performance framework for Zero-Knowledge Machine Learning (ZKML) circuits compiled via EZKL (Halo2 proof system backend).

### **Abstract:**

Evaluating the resource cost (proving time, peak RAM, and proof size) of Zero-Knowledge Machine Learning (ZKML) remains a major barrier to deploying verifiable AI. Traditional cost estimations assume linear additivity, where total prover time equals the sum of individual layer latencies.

This repository provides an empirical analysis of ZKML circuit compilation using EZKL/Halo2. By isolating dense matrix multiplications, fixed-point lookup tables (LUTs), spatial convolutions, and multi-layer CNN/Transformer pipelines, we demonstrate that ZKML proving cost is strictly non-additive at the layer level. Instead, proving time is governed by the underlying cryptographic matrix constraints: discrete grid bounds, active grid cell density, non-linear lookup spans, static weight commitments, and Fast Fourier Transform (FFT) domain complexity.

We present a non-linear, zero-intercept cryptographic cost model evaluated across 50 benchmark runs. The model achieves an overall Mean Absolute Percentage Error (MAPE) of **12.69%** across distinct architectures (including deep Convolutional Neural Networks and Multi-Head Attention Transformer blocks) and demonstrates strong out-of-sample stability (**13.60% ± 4.10% MAPE**) under Repeated 5-Fold Cross-Validation.

### **Repository Structure:**

```text
zkml-cost-model-benchmarks/
│
├── README.md                          # Project and results write-up
├── requirements.txt                   # Environment dependencies (PyTorch, EZKL, scikit-learn, etc.)
├── .gitignore                         # Excludes temporary cache files (
│
├── benchmarks/                        # Python scripts that execute EZKL circuit compilations
│   ├── benchmark_linear_only.py       # Single-layer Linear sweeps
│   ├── benchmark_rigorous_suite.py    # Multi-repeat baseline sweeps
│   ├── benchmark_transformers.py      # Transformer layer sweeps
│   ├── benchmark_transformer_block.py # Full Transformer Block benchmarks
│   ├── benchmark_logrows_sweep.py     # Channel sweeps up to logrows threshold (OOM crash script)
│   └── benchmark_blind_test.py        # Randomised test suite for validation
│
├── cluster_scripts/                   # SGE (.sh) batch submission scripts for Eddie HPC
│   ├── run_benchmark_conv2d.sh
│   ├── run_benchmark_lsc2d_repeat.sh
│   ├── run_benchmark_rigorous_suite.sh
│   ├── run_benchmark_transformers.sh
│   ├── run_benchmark_transformer_block.sh
│   ├── run_benchmark_sweep.sh
│   └── run_blind_test.sh
│
├── data/                              # Benchmark result CSV datasets
│   ├── circuit_features_master.csv    # Extracted Plonkish circuit metrics (A_total, L_span, etc.)
│   ├── deep_validation_results.csv    # Extended validation architectures
│   ├── high_res_rigorous_results.csv  # High-resolution CNN channel sweeps
│   ├── transformer_block_results.csv  # Transformer benchmark dataset
│   └── blind_test_results.csv         # Blind test evaluation dataset
│
└── analysis/                          # Mathematical fitting & cross-validation scripts
    ├── fit_cost_model.py              # Phase 1-5: Early layer-additive OLS model
    ├── analyze_blind_test.py          # Delta-model evaluation script
    ├── analyze_final_physics.py       # Phase 9: Pure NNLS zero-intercept model
    ├── analyze_5fold_cv.py            # 5-fold cross-validation script
    ├── analyze_loao.py                # Leave-One-Architecture-Out zero-shot generalization script
    ├── analyze_cryptographic_math.py  # Final O(N log N) FFT model + 5-Fold CV
    ├── analyze_detailed_breakdown.py  # Detailed per-model actual vs. predicted itemization
    └── analyze_rigorous_cv.py         # Repeated K-Fold & Leave-One-Config-Out (scikit-learn)
```
