## **Predictive Cost Modelling and Empirical Characterisation of ZKML Circuits:**

An empirical benchmark suite and predictive performance framework for Zero-Knowledge Machine Learning (ZKML) circuits compiled via EZKL (Halo2 proof system backend).

### **Abstract:**

Evaluating the resource cost (proving time, peak RAM, and proof size) of Zero-Knowledge Machine Learning (ZKML) remains a major barrier to deploying verifiable AI. Traditional cost estimations assume linear additivity, where total prover time equals the sum of individual layer latencies.

This repository provides an empirical analysis of ZKML circuit compilation using EZKL/Halo2. By isolating dense matrix multiplications, fixed-point lookup tables (LUTs), spatial convolutions, and multi-layer CNN/Transformer pipelines, we demonstrate that ZKML proving cost is strictly non-additive at the layer level. Instead, proving time is governed by the underlying cryptographic matrix constraints: discrete grid bounds, active grid cell density, non-linear lookup spans, static weight commitments, and Fast Fourier Transform (FFT) domain complexity.

We present a non-linear, zero-intercept cryptographic cost model evaluated across 50 benchmark runs. The model achieves an overall Mean Absolute Percentage Error (MAPE) of **12.69%** across distinct architectures (including deep Convolutional Neural Networks and Multi-Head Attention Transformer blocks) and demonstrates strong out-of-sample stability (**13.60% ± 4.10% MAPE**) under Repeated 5-Fold Cross-Validation.

Finally, we apply this validated cost model to a real-world engineering problem: constructing an empirical Pareto frontier that maps cryptographic proving time against quantised model accuracy on the MNIST dataset. By evaluating 8 distinct neural network architectures across 5 bit-width quantization scales (4-bit to 16-bit), we define the optimal network architectures (depth vs. width) for ZK circuits, and identify a 12-bit precision optimal operating point that reduces proving overhead for non-linear activations by 66% with zero accuracy loss.

### **Repository Structure:**

```text
zkml-predictive-cost-model/
│
├── README.md                          # Project and results write-up
├── requirements.txt                   # Environment dependencies (PyTorch, EZKL, scikit-learn, etc.)
├── .gitignore                         # Excludes temporary cache files, HPC logs, and EZKL artefacts
├── pareto_frontier_final.png          # Empirical Pareto frontier visualisation 
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
│   ├── compile_quantised_models.py    # Compiles Pareto models across 5 bit-width scales (4 to 16-bit)
│   ├── deep_validation_benchmark.py
│   ├── evaluate_true_quantised_accuracies.py # Simulates EZKL quantisation to extract true test accuracy
│   ├── extract_and_validate_blind.py
│   ├── extract_circuit_features.py    # Extracts A_total, L_span, and C_size from circuits
│   └── fetch_srs.py
│   └── train_mnist_models.py          # Trains the 8 baseline architectures on MNIST
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
│   ├── run_compile_quantised.sh
│   ├── run_deep_validation.sh
│   ├── run_extract_and_validate_blind.sh
│   └── run_extract_circuit_features.sh
│   ├── run_train_mnist.sh             
│   └── run_true_accuracies.sh      
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
│   └── true_accuracies.txt            # Test accuracies for the 40 quantised configurations
│
└── analysis/                          # Mathematical fitting & cross-validation scripts
    ├── analyse_5fold_cv.py            # 5-fold cross-validation script
    ├── analyse_blind_test.py          # Delta-model evaluation script
    ├── analyse_cryptographic_math.py  # Final O(N log N) FFT model + 5-Fold CV
    ├── analyse_deep_val.py
    ├── analyse_final_physics.py       # Phase 9: Pure NNLS zero-intercept model
    ├── analyse_hybrid_model.py
    ├── analyse_loao.py                # Leave-One-Architecture-Out zero-shot generalisation script
    ├── analyse_rigorous_cv.py         # Repeated K-Fold & Leave-One-Config-Out (scikit-learn)
    ├── analyse_rigorous_results.py
    ├── analyse_rigorous_results_delta.py
    ├── analyse_step_function.py
    ├── analyse_transformers.py
    ├── analyse_true_hybrid.py
    ├── evaluate_pareto_costs.py       # Predicts proving times for the 40 Pareto configuration
    ├── fit_cost_model.py              # Phase 1-5: Early layer-additive OLS model
    ├── plot_activation_results.py
    ├── plot_combinatorial_results.py
    ├── plot_pareto_frontier.py        # Generates the final empirical Pareto frontier visualisation
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

## Model Validation

To ensure the model is mathematically stable and to prevent overfitting, we conducted three validation stress tests using `analysis/analyse_rigorous_cv.py`, `analysis/analyse_loao.py`, and `analysis/analyse_cryptographic_math.py`.

### 1. Repeated 5-Fold Cross-Validation (10 Seeds / 50 Splits)

Executed via `analysis/analyse_rigorous_cv.py`, this test randomises the dataset across 50 distinct train/test splits to verify coefficient stability.

* **Mean Out-of-Sample MAPE:** **13.60%**
* **Standard Deviation:** **4.10%**
* **Min / Max Split Range:** **[6.48% – 25.84%]**

> **Conclusion:** The narrow standard deviation demonstrates tight coefficient convergence. Regardless of the training subset, the formula consistently identifies the universal cryptographic constants of the EZKL backend rather than memorising dataset noise.

### 2. Leave-One-Config-Out (LOCO) Validation

Executed via `analysis/analyse_rigorous_cv.py`, this test forces the model to predict proving times for an entire configuration size (e.g., all `Size 32` networks) that was strictly excluded from its training data.

* **Mean Blind MAPE:** **20.20%**

> **Conclusion:** Predicting compiler scaling on unseen dimensions is difficult due to the discrete $2^k$ row step-functions. Achieving ~20% error on blind configurations proves the formula correctly evaluates the underlying physical matrix constraints ($A_{\text{total}}$, $L_{\text{span}}$) to calculate execution costs, rather than relying on historical runtime memorisation.

### 3. Leave-One-Architecture-Out (LOAO) Zero-Shot Cross-Prediction

Executed via `analysis/analyse_loao.py`, this test isolates entire architectural families to prove the necessity of hybrid benchmarking datasets.

* **Train on CNNs $\to$ Predict Transformers:** **41.25% MAPE**
* **Train on Transformers $\to$ Predict CNNs:** **46.99% MAPE**

> **Conclusion:** ZK cost profiling requires diverse benchmarks. CNNs are constrained by dense matrix assignments ($A_{\text{total}}$), whereas Transformers are bound by non-linear lookups ($L_{\text{span}}$). Training on a single architecture causes "feature starvation," rendering it incapable of calculating the alternative constraint.

### 4. Asymptotic Accuracy at Scale

Extracted from the itemised predictions in `analysis/analyse_cryptographic_math.py`, the model's accuracy improves dramatically as the neural network size increases.

* **Large CNNs ($>300$s proving time):** The error rate drops to between **1.5% and 3.0%**. At low compute scales, fixed system overheads (OS background noise, memory allocation) skew percentage errors. At scale, this noise becomes mathematically negligible, and proving time strictly adheres to the derived $\mathcal{O}(N \log N)$ FFT complexity.
* **Transformer "Lookup Spike":** Testing solely on Multi-Head Attention blocks yielded an **8.08% average error**. The $L_{\text{span}}$ variable successfully isolated the massive computational penalty of non-linear GELU/Softmax tables that breaks traditional parameter-counting ML cost models.

---

### **Empirical Benchmark Results (The Roadmap to the Model):**

All experiments were executed on the University of Edinburgh's Eddie HPC Cluster using dedicated compute nodes running Linux CentOS 7, Python 3.10, PyTorch 2.13.0+cu130, and EZKL 23.0.5 (Halo2 backend).

### Phase 1: The Non-Additivity Discovery

*Generated via `benchmarks/benchmark_linear_only.py` and `benchmarks/benchmark_linear_relu.py` (Dataset: `data/linear_only_results.csv`)*

Comparing an isolated `Linear` layer (1 -> N outputs) against a fused `Linear + ReLU + Linear` "sandwich" model demonstrates that proving overhead is inherently non-additive.

| Output Dimension ($N$) | Fused Model Time | Isolated `Linear` Time | `logrows` ($k$) | Proof Size |
| --- | --- | --- | --- | --- |
| **2,000** | 30.74s | 38.04s | 17 | 0.58 MB |
| **4,000** | 53.63s | 60.93s | 17 | 1.14 MB |
| **8,000** | 77.10s | 88.86s | 17 | 2.26 MB |
| **10,000** | 78.73s | 85.76s | 17 | 2.84 MB |

> **Key Finding:** Fusing `ReLU` and a second `Linear` layer incurs **zero additional proving time penalty**. Graph optimisation and horizontal column packing allow composite arithmetic operations to share the same grid bounds (2^17 = 131,072 rows).

---

### Phase 2: Non-Linear Lookup Tables (`Sigmoid`) vs. Arithmetic

*Generated via `benchmarks/benchmark_sigmoid.py` (Dataset: `data/sigmoid_results.csv`)*

Non-linear operations require fixed-point lookup tables (LUTs) in Halo2. Sweeping a single `Sigmoid` layer up to 10,000 elements reveals a distinct trade-off between time complexity and proof payload space.

| Input Elements ($N$) | `Linear` Time | `Sigmoid` Time | `Linear` Proof Size | `Sigmoid` Proof Size |
| --- | --- | --- | --- | --- |
| **2,000** | 38.04s | 37.32s | 0.30 MB | 0.58 MB |
| **4,000** | 60.93s | 37.31s | 0.59 MB | 1.14 MB |
| **8,000** | 88.86s | 38.01s | 1.17 MB | 2.26 MB |
| **10,000** | 85.76s | 67.18s | 1.45 MB | 2.84 MB |

> **Key Finding:** `Sigmoid` lookup tables exhibit lower proving latency at small-to-medium scales because element-wise lookups avoid dense cross-multiplication. However, non-linear lookup tables **double the final proof size** due to the requirement of embedding table commitments.

---

### Phase 3: Spatial Convolutions (`Conv2d`) & Column Packing

*Generated via `benchmarks/benchmark_conv2d.py` (Dataset: `data/conv2d_results.csv`)*

Sweeping output channel count ($C_{\text{out}} \in [4, 60]$) on a fixed $16 \times 16$ RGB spatial input ($3 \times 16 \times 16$) demonstrates how horizontal column expansion produces step jumps in proving time.

| Channels ($C_{\text{out}}$) | `logrows` ($k$) | Proving Time | Time Increment |
| --- | --- | --- | --- |
| **4 – 8** | 17 | ~41.9s | Baseline |
| **12 – 20** | 17 | ~74.8s | +32.9s |
| **24 – 28** | 17 | ~113.6s | +38.8s |
| **32 – 40** | 17 | ~141.4s | +27.8s |
| **44 – 52** | 17 | ~176.6s | +35.2s |
| **56 – 60** | 17 | ~210.1s | +33.5s |

> **Key Finding:** Proving time grows in discrete **~33–35 second steps**. Because `logrows` remains constant at 17, each step jump corresponds to EZKL allocating an additional column to the Halo2 constraint matrix to pack the operations horizontally.

---

### Phase 4: Composite CNN Pipeline & Downsampling

*Generated via `benchmarks/benchmark_rigorous_suite.py` (Dataset: `data/high_res_rigorous_results.csv`)*

Benchmarking a full multi-layer CNN pipeline (`Conv2d -> ReLU -> MaxPool2d -> Linear`) highlights the effect of spatial downsampling on proof size.

| Output Channels | Pipeline Proving Time | Isolated `Conv2d` Time | Pipeline Proof Size | Isolated `Conv2d` Proof Size |
| --- | --- | --- | --- | --- |
| **8** | 75.73s | 41.90s | **0.13 MB** | 0.41 MB |
| **16** | 127.04s | 74.64s | **0.15 MB** | 0.71 MB |
| **24** | 183.95s | 111.91s | **0.16 MB** | 1.01 MB |

> **Key Finding:** While the composite pipeline adds proving latency due to additional operations, `MaxPool2d` spatial downsampling ($16 \times 16 \to 8 \times 8$) contracts the final output commitment space, **reducing total proof payload size by over 80%**.

---

### Phase 5: Layer-Summation Failure & The Blind Test

*Evaluated via `analysis/analyse_blind_test.py` (Dataset: `data/blind_test_results.csv`)*

Before analysing circuit compilation directly, we attempted to predict multi-layer model times by summing individual layer latencies and applying a flat "discount delta" (`Time = 0.68 * Sum(Layers)`) to account for compiler fusion. We evaluated this model on an unseen "Blind Test" dataset.

| Model Configuration | Actual Proving Time | Predicted Time (Delta Model) | Error (%) | Impact / Finding |
| --- | --- | --- | --- | --- |
| **Linear Fused (N=4000)** | 53.63s | 72.93s | **36.0%** | Overestimated due to layer fusion |
| **Sigmoid Sweep (N=8000)** | 38.01s | 65.37s | **72.0%** | Massive over-prediction on LUTs |
| **Conv2D Pipeline (C=16)** | 127.04s | 145.46s | **14.5%** | Moderate error on CNNs |
| **Blind Test Overall** | — | — | **33.82% MAPE** | **Failure:** Flat deltas cannot model dynamic graph fusion. |

> **Key Finding:** Compiler graph fusion is highly dynamic, not a flat percentage. Traditional ML models built on PyTorch/ONNX layer counts fundamentally fail because they cannot predict how EZKL packs operations into the underlying physical grid.

---

### Phase 6: Hitting the Hardware Wall (The $2^{18}$ OOM Crash)

*Generated via `benchmarks/benchmark_logrows_sweep.py` (Dataset: `data/logrows_sweep_results.csv`, Cluster Script: `cluster_scripts/run_benchmark_sweep.sh`)*

To test the physical limits of `logrows` ($k$), we executed a high-resolution channel sweep on a fixed $16 \times 16$ RGB spatial input ($3 \times 16 \times 16$) until the system failed.

| Channels ($C_{\text{out}}$) | Total Rows | Grid Degree ($k$) | Actual Proving Time | Peak Memory / Status |
| --- | --- | --- | --- | --- |
| **4 – 8** | ~45,000 | 17 ($2^{17} = 131,072$) | 41.90s | ~18 GB RAM (Passed) |
| **12 – 20** | ~85,000 | 17 ($2^{17} = 131,072$) | 74.84s | ~32 GB RAM (Passed) |
| **24** | ~122,000 | 17 ($2^{17} = 131,072$) | 113.60s | ~58 GB RAM (Passed) |
| **26+** | **>131,072** | **18 ($2^{18} = 262,144$)** | **FAILED** | **OOM CRASH (256GB Node)** |

> **Key Finding:** ZK hardware scaling is strictly non-continuous. Crossing the threshold of 131,072 rows forced $k \to 18$, instantly doubling the evaluation domain. This overwhelmed the 256GB RAM limit on the Eddie HPC node, proving that cost models **must account for discrete 2^k grid step-functions**.

---

### Phase 7: Isolating Non-Linear Transformers & GELU Lookup Spikes

*Generated via `benchmarks/benchmark_transformer_block.py` and `benchmarks/benchmark_transformers.py` (Dataset: `data/transformer_block_results.csv`, Evaluated via `analysis/analyse_transformers.py`)*

To capture both linear and non-linear bottlenecks, we expanded benchmarks to include Transformer blocks, comparing them against CNNs of equivalent parameter counts.

| Model Architecture | Parameter Size ($d_{\text{model}}$) | Primary Bottleneck | Actual Proving Time | Proof Size |
| --- | --- | --- | --- | --- |
| **MiniCNN** | Size 16 | Matrix Mult (`A_total`) | **108.84s** | 0.71 MB |
| **TransformerBlock** | Size 16 | GELU/Softmax (`L_span`) | **276.73s** | 2.14 MB |
| **MiniCNN** | Size 32 | Matrix Mult (`A_total`) | **200.54s** | 1.42 MB |
| **TransformerBlock** | Size 32 | GELU/Softmax (`L_span`) | **416.97s** | 4.88 MB |

> **Key Finding:** Transformers take **more than double the proving time** of CNNs at equivalent parameter scales due to massive fixed-point lookup table spans ($L_{\text{span}}$). Cost models must extract both grid density and lookup table spans directly from the compiled circuit.

---

### Phase 8: Circuit Feature Extraction & Ridge L2 Regularisation Failure

*Extracted via `benchmarks/extract_circuit_features.py` (Dataset: `data/circuit_features_master.csv`, Evaluated via `analysis/train_cost_model.py`)*

Moving away from PyTorch layers, we extracted raw circuit features ($A_{\text{total}}, L_{\text{span}}, C_{\text{size}}, D_{\text{size}}$) and fitted a Ridge (L2 Penalised) Regression model with a fixed intercept ($R_{\text{base}}$):

$$\text{Predicted Time} = R_{\text{base}} + w_1 A_{\text{total}} + w_2 L_{\text{span}} + w_3 C_{\text{size}} + w_4 D_{\text{size}}$$

| Model Test Run | Parameter Size | Actual Proving Time | Ridge Predicted Time | Error (%) |
| --- | --- | --- | --- | --- |
| **MiniCNN** | Size 14 | 141.66s | 74.78s | **47.21%** |
| **MiniCNN** | Size 22 | 285.84s | 116.71s | **59.17%** |
| **MiniCNN** | Size 30 | 287.43s | 131.94s | **54.10%** |
| **TransformerBlock** | Size 18 | 289.79s | 161.21s | **44.37%** |
| **Overall Dataset** | — | — | — | **51.21% MAPE** |

> **Why it Failed:** Because we forced an intercept ($R_{\text{base}}$) and applied L2 regularisation, the penalty acted like a rubber band—dragging all large prediction weights down by exactly 50%.

---

### Phase 9: The Zero-Intercept Pure Physics Model (NNLS)

*Evaluated via `analysis/analyse_final_physics.py`*

We stripped out the L2 penalty and artificial intercept, applying unpenalised Non-Negative Least Squares (NNLS) directly to pure circuit features:

$$\widehat{T}_{\text{prove}} = (1.51 \times 10^{-4}) \cdot A_{\text{total}} + (1.00 \times 10^{-3}) \cdot L_{\text{span}} + (6.49 \times 10^{-2}) \cdot C_{\text{size}}$$

| Evaluation Dataset | Primary Feature Tested | Overall Dataset MAPE | Key Observation |
| --- | --- | --- | --- |
| **`transformer_block_results.csv`** | Non-Linear Lookups ($L_{\text{span}}$) | **9.45%** | Successfully isolated GELU lookup penalties |
| **`high_res_rigorous_results.csv`** | Dense Grid Assignments ($A_{\text{total}}$) | **11.41%** | Sub-5% error on large CNNs ($S \ge 32$) |
| **`deep_validation_results.csv`** | Unseen Topologies | **22.71%** | Higher error on small discrete models |
| **5-Fold Cross-Validation** | Randomised Splits | **12.56%** | High coefficient stability across folds |

> **Key Finding:** Removing artificial baseline intercepts forced the solver to mirror the circuit's true physical properties, cutting overall prediction errors from ~51% down to ~12.5%.

---

### Phase 10: Cross-Architecture Stress Test (LOAO)

*Evaluated via `analysis/analyse_loao.py`*

To test whether a model trained on one network family could predict another, we performed Leave-One-Architecture-Out (LOAO) zero-shot validation across CNNs and Transformers.

| Training Family | Unseen Test Family | Zero-Shot MAPE | Failure Mode Analysis |
| --- | --- | --- | --- |
| **43 CNN Models** | **7 Transformers** | **41.25%** | CNNs lack lookups; model under-predicted GELU penalties |
| **7 Transformers** | **43 CNN Models** | **46.99%** | Over-penalised lookups; over-predicted dense matrix maths |

> **Key Finding:** Plonkish arithmetisation is a **multi-bottleneck system**. Training data *must* contain both dense maths (CNNs) and lookup tables (Transformers) for the solver to accurately balance the coefficients.

---

### Phase 11: Final Cryptographic Non-Linear Model & Asymptotic Scaling

*Evaluated via `analysis/analyse_cryptographic_math.py` and `analysis/analyse_cryptographic_math.py`*

Grounding the equation in the actual algorithmic complexity of Halo2's backend, Fast Fourier Transforms ($O(N \log N)$) and Multi-Scalar Multiplications ($O(N)$), yielded our final published equation:

$$\widehat{T}_{\text{prove}} \approx (2.94 \times 10^{-8}) \cdot (D_{\text{size}} \log_2 D_{\text{size}}) + (1.51 \times 10^{-4}) \cdot A_{\text{total}} + (9.77 \times 10^{-4}) \cdot L_{\text{span}} + (6.63 \times 10^{-2}) \cdot C_{\text{size}}$$

#### Itemised Accuracy Breakdown Across All Architectures

| Model Family | Size / Config | Actual Time | Predicted Time | Difference | Error (%) |
| --- | --- | --- | --- | --- | --- |
| **MiniCNN** | Size 8 | 62.27s | 96.39s | +34.12s | **54.79%** (Noise at low scale) |
| **MiniCNN** | Size 16 | 108.84s | 129.65s | +20.81s | **19.13%** |
| **MiniCNN** | Size 32 | 218.56s | 214.43s | -4.13s | **1.89%** |
| **MiniCNN** | Size 40 | 254.58s | 256.92s | +2.34s | **0.92%** |
| **MiniCNN** | Size 56 | 339.54s | 345.10s | +5.56s | **1.64%** |
| **MiniCNN** | Size 64 | 395.99s | 387.93s | -8.06s | **2.03%** |
| **TransformerBlock** | Size 16 | 276.73s | 301.62s | +24.89s | **9.00%** |
| **TransformerBlock** | Size 18 | 289.79s | 292.55s | +2.76s | **0.95%** |
| **TransformerBlock** | Size 32 | 417.83s | 389.49s | -28.34s | **6.78%** |

#### Rigorous Validation Summary

| Validation Method | Total Iterations | Evaluated Metric | Result |
| --- | --- | --- | --- |
| **Repeated 5-Fold CV** | 10 Seeds (50 Splits) | Out-of-Sample Mean MAPE | **13.60% ($\pm 4.10\%$)** |
| **Leave-One-Config-Out** | 18 Configuration Groups | Strict Unseen Group MAPE | **20.20%** |
| **Asymptotic Large-Scale** | Models $S \ge 32$ ($T > 200\text{s}$) | Asymptotic Proving MAPE | **1.5% – 3.0%** |

> **Final Conclusion:** At low scales, fixed system setup noise causes higher percentage errors. But as the neural network grows and fills the grid, noise vanishes, and proving time **strictly obeys our $O(N \log N)$ cryptographic equation**.

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

> **Note on File Paths for Execution:** During the original HPC batch processing, these scripts were executed in a flat directory structure. If you are reproducing this research locally using this structured repository, please note that some analysis scripts may contain local file path references (e.g., `pd.read_csv('circuit_features_master.csv')`). To run them successfully, either execute them directly from within the `data/` folder or quickly update the pandas read path in the script to `../data/[filename].csv`.


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

To run Repeated K-Fold and Leave-One-Config-Out (LOCO) cross-validation:

```bash
python analysis/analyse_rigorous_cv.py

```

### 4. Submitting Batch Jobs on Sun Grid Engine (HPC)

To submit benchmark runs to an SGE cluster (like the University of Edinburgh's Eddie node):

```bash
qsub cluster_scripts/run_benchmark_conv2d.sh

```
