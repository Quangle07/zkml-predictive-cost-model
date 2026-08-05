## **Predictive Cost Modelling and Empirical Characterisation of ZKML Circuits:**

An empirical benchmark suite and predictive performance framework for Zero-Knowledge Machine Learning (ZKML) circuits compiled via EZKL (Halo2 proof system backend).

### **Abstract:**

Evaluating the resource cost (proving time, peak RAM, and proof size) of Zero-Knowledge Machine Learning (ZKML) remains a major barrier to deploying verifiable AI. Traditional cost estimations assume linear additivity, where total prover time equals the sum of individual layer latencies.

This repository provides an empirical analysis of ZKML circuit compilation using EZKL/Halo2. By isolating dense matrix multiplications, fixed-point lookup tables (LUTs), spatial convolutions, and multi-layer CNN/Transformer pipelines, we demonstrate that ZKML proving cost is strictly non-additive at the layer level. Instead, proving time is governed by the underlying cryptographic matrix constraints: discrete grid bounds, active grid cell density, non-linear lookup spans, static weight commitments, and Fast Fourier Transform (FFT) domain complexity.

We present a non-linear, zero-intercept cryptographic cost model evaluated across 50 benchmark runs. The model achieves an overall Mean Absolute Percentage Error (MAPE) of **12.69%** across distinct architectures (including deep Convolutional Neural Networks and Multi-Head Attention Transformer blocks) and demonstrates strong out-of-sample stability (**13.60% ± 4.10% MAPE**) under Repeated 5-Fold Cross-Validation.

### **Repository Structure:**

```text
zkml-predictive-cost-model/
│
├── README.md                          # Project and results write-up
├── requirements.txt                   # Environment dependencies (PyTorch, EZKL, scikit-learn, etc.)
├── .gitignore                         # Excludes temporary cache files, HPC logs, and EZKL artifacts
│
├── benchmarks/                        # Python scripts that execute EZKL circuit compilations
│   ├── benchmark_activation_complete.py
│   ├── benchmark_blind_test.py        # Randomised test suite for validation
│   ├── benchmark_cnn.py
│   ├── benchmark_conv2d.py
│   ├── benchmark_linear_only.py       # Single-layer Linear sweeps
│   ├── benchmark_linear_relu.py
│   ├── benchmark_logrows_sweep.py     # Channel sweeps up to logrows threshold (OOM crash script)
│   ├── benchmark_operators.py
│   ├── benchmark_quantisation.py
│   ├── benchmark_rigorous_suite.py    # Multi-repeat baseline sweeps
│   ├── benchmark_sigmoid.py
│   ├── benchmark_transformer_block.py # Full Transformer Block benchmarks
│   ├── benchmark_transformers.py      # Transformer layer sweeps
│   ├── deep_validation_benchmark.py
│   ├── extract_and_validate_blind.py
│   ├── extract_circuit_features.py    # Extracts A_total, L_span, and C_size from circuits
│   └── fetch_srs.py
│
├── cluster_scripts/                   # SGE (.sh) batch submission scripts for Eddie HPC
│   ├── run_benchmark_activation.sh
│   ├── run_benchmark_cnn.sh
│   ├── run_benchmark_conv2d.sh
│   ├── run_benchmark_linear_only.sh
│   ├── run_benchmark_linear_relu.sh
│   ├── run_benchmark_operators.sh
│   ├── run_benchmark_quantisation.sh
│   ├── run_benchmark_rigorous_suite.sh
│   ├── run_benchmark_sigmoid.sh
│   ├── run_benchmark_sweep.sh
│   ├── run_benchmark_transformer_block.sh
│   ├── run_benchmark_transformers.sh
│   ├── run_blind_test.sh
│   ├── run_deep_validation.sh
│   ├── run_extract_and_validate_blind.sh
│   └── run_extract_circuit_features.sh
│
├── data/                              # Benchmark result CSV datasets & output logs
│   ├── activation_results_complete.csv
│   ├── all_quantisation_results.csv
│   ├── analyse_blind_test_results.txt
│   ├── analyse_transformers_results.txt
│   ├── blind_test_results.csv         # Blind test evaluation dataset
│   ├── circuit_features_master.csv    # Extracted Plonkish circuit metrics
│   ├── cnn_results.csv
│   ├── combinatorial_results.json
│   ├── conv2d_results.csv
│   ├── deep_validation_results.csv    # Extended validation architectures
│   ├── fit_cost_model_results.txt
│   ├── high_res_rigorous_results.csv  # High-resolution CNN channel sweeps
│   ├── linear_only_results.csv
│   ├── logrows_sweep_results.csv
│   ├── operator_benchmark_results.csv
│   ├── sigmoid_results.csv
│   ├── train_cost_model_results.txt
│   ├── transformer_benchmark_results.csv
│   └── transformer_block_results.csv  # Transformer benchmark dataset
│
└── analysis/                          # Mathematical fitting & cross-validation scripts
    ├── analyse_5fold_cv.py            # 5-fold cross-validation script
    ├── analyse_blind_test.py          # Delta-model evaluation script
    ├── analyse_cryptographic_math.py  # Final O(N log N) FFT model + 5-Fold CV
    ├── analyse_deep_val.py
    ├── analyse_final_physics.py       # Phase 9: Pure NNLS zero-intercept model
    ├── analyse_hybrid_model.py
    ├── analyse_loao.py                # Leave-One-Architecture-Out zero-shot generalization script
    ├── analyse_rigorous_cv.py         # Repeated K-Fold & Leave-One-Config-Out (scikit-learn)
    ├── analyse_rigorous_results.py
    ├── analyse_rigorous_results_delta.py
    ├── analyse_step_function.py
    ├── analyse_transformers.py
    ├── analyse_true_hybrid.py
    ├── fit_cost_model.py              # Phase 1-5: Early layer-additive OLS model
    ├── plot_activation_results.py
    ├── plot_combinatorial_results.py
    └── train_cost_model.py
```

### **Theoretical Framework:** 

#### 1. Notation Guide:

To predict computational cost, machine learning operations are mapped directly to the physical layout of a PLONKish Arithmetisation matrix (the cryptographic grid executed by the Halo2 prover).

* **$k$ (`logrows`):** The base-2 exponent defining the vertical row capacity of the cryptographic grid ($N = 2^k$).
* **$D_{\text{size}}$ (`domain_size`):** The total row capacity of the allocated grid, equal to $2^k$.
* **$A_{\text{total}}$ (`total_assignments`):** The total active cell assignments populated within the constraint matrix (Grid Density).
* **$L_{\text{span}}$ (`lookup_span`):** The total grid footprint of fixed-point lookup tables required for non-linear activations (e.g., Sigmoid, GELU, ReLU).
* **$C_{\text{size}}$ (`total_const_size`):** The constraint overhead required to commit the network's static parameters (weights and biases) into the cryptographic grid.
* **$w_{\text{fft}}, w_{\text{msm}}, w_{\text{assign}}, w_{\text{lookup}}, w_{\text{const}}$:** Time coefficients representing the actual seconds required to compute one unit of each respective variable.

#### 2. The Failure of Layer-Based Additivity

Standard ML models assume $Cost(A + B) = Cost(A) + Cost(B)$. In ZKML, compiler graph optimisation (fusing adjacent layers like `Conv2D` $\to$ `ReLU`) and horizontal column packing mean that adding layers often incurs zero additional proving penalty, rendering standard linear regression on PyTorch layer types (`analysis/fit_cost_model.py`) highly inaccurate.

#### 3. Cryptographic Grid Bounds ($2^k$)

EZKL compiles ONNX compute graphs into Halo2 matrices where capacity scales strictly in powers of two ($D_{\text{size}} = 2^k$). When circuit constraints exceed $2^k$, the compiler increments to $k+1$, doubling the evaluation domain. This non-continuous scaling causes massive, discrete step-function spikes in peak RAM usage and prover time.

#### 4. Algorithmic Complexity (FFTs & MSMs)

Prover execution in Halo2 is dominated by two primary cryptographic primitives over the evaluation domain $N = 2^k$:

* **Fast Fourier Transforms (FFTs):** Scaling as $\mathcal{O}(N \log_2 N)$.
* **Multi-Scalar Multiplications (MSMs):** Scaling as $\mathcal{O}(N)$.

#### 5. The Unified Cryptographic Cost Model

We model the estimated proving time ($\widehat{T}_{\text{prove}}$) using a non-linear algebraic equation that maps these exact physical constraints and cryptographic boundaries:

$$\widehat{T}_{\text{prove}} = w_{\text{fft}} \cdot (D_{\text{size}} \log_2 D_{\text{size}}) + w_{\text{msm}} \cdot D_{\text{size}} + w_{\text{assign}} \cdot A_{\text{total}} + w_{\text{lookup}} \cdot L_{\text{span}} + w_{\text{const}} \cdot C_{\text{size}}$$

---

## Model Fitting & Parameter Identification

Using Non-Negative Least Squares (NNLS) optimisation via `analysis/analyse_cryptographic_math.py`, we extracted the fundamental time coefficients for the Halo2/EZKL backend. The model was fitted on a merged dataset of 50 benchmark runs comprising both CNNs and Transformer blocks (aggregated from `data/high_res_rigorous_results.csv`, `data/transformer_block_results.csv`, and `data/deep_validation_results.csv`).

### Fitted Operational Parameters

| Parameter | Feature Name | Mathematical Term | Fitted Value | Physical Meaning / Description |
| --- | --- | --- | --- | --- |
| **$w_{\text{fft}}$** | `fft_work` | $D_{\text{size}} \log_2 D_{\text{size}}$ | **$2.9350 \times 10^{-8}\text{ s}$** | FFT processing time per domain element unit |
| **$w_{\text{msm}}$** | `msm_work` | $D_{\text{size}}$ | **$0.0000 \times 10^{0}\text{ s}$** | Linear MSM overhead (zeroed due to FFT $\mathcal{O}(N \log N)$ dominance) |
| **$w_{\text{assign}}$** | `total_assignments` | $A_{\text{total}}$ | **$1.5115 \times 10^{-4}\text{ s}$** | Time per populated active cell assignment (Grid Density) |
| **$w_{\text{lookup}}$** | `lookup_span` | $L_{\text{span}}$ | **$9.7656 \times 10^{-4}\text{ s}$** | Time penalty per non-linear lookup table entry |
| **$w_{\text{const}}$** | `total_const_size` | $C_{\text{size}}$ | **$6.6258 \times 10^{-2}\text{ s}$** | Time penalty per static parameter weight commitment |

> **Note on $w_{\text{msm}} = 0$:** Because $D_{\text{size}} \log_2 D_{\text{size}}$ and $D_{\text{size}}$ scale concurrently with grid height $k$, NNLS identified FFT complexity $\mathcal{O}(N \log_2 N)$ as the primary driver of domain latency, zeroing out $w_{\text{msm}}$ to eliminate collinear redundancy. 

### Definitive Predictive Formula

$$\widehat{T}_{\text{prove}} \approx (2.94 \times 10^{-8}) \cdot (D_{\text{size}} \log_2 D_{\text{size}}) + (1.51 \times 10^{-4}) \cdot A_{\text{total}} + (9.77 \times 10^{-4}) \cdot L_{\text{span}} + (6.63 \times 10^{-2}) \cdot C_{\text{size}}$$

---



---

## Reproduction Instructions:

### 1. Environment Setup

```bash

git clone https://github.com/Quangle07/zkml-predictive-cost-model.git
cd zkml-predictive-cost-model

# Activate Conda environment
conda create -n ezkl_env python=3.10 -y
conda activate ezkl_env

# Install dependencies
pip install -r requirements.txt

```

### 2. Running Benchmarks Locally

To execute a local single-layer benchmark run (e.g., sweeping `Linear` layers):

```bash
python benchmarks/benchmark_linear_only.py

```

### 3. Running Analysis & Cost Model Fitting

To fit the final cryptographic non-linear model ($O(N \log N)$ FFTs) on the pre-collected datasets:

```bash
python analysis/analyse_cryptographic_math.py

```

To view the detailed per-model itemization (Actual vs. Predicted):

```bash
python analysis/analyse_detailed_breakdown.py

```

To run Repeated K-Fold and Leave-One-Config-Out (LOCO) cross-validation:

```bash
python analysis/analyse_rigorous_cv.py

```

### 4. Submitting Batch Jobs on Sun Grid Engine (HPC)

To submit benchmark runs to an SGE cluster (like the University of Edinburgh's Eddie node):

```bash
qsub cluster_scripts/run_benchmark_conv2d.sh

```
