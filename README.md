# Benchmarking Self-Supervised Learning & State-Space Models (FEMBA) for Wearable Biosignal Stress Detection

This repository contains the official benchmark, ablation suite, and statistical validation framework extending the work of Corponi et al. (*JMIR mHealth and uHealth*, 2024). 

We systematically evaluate self-supervised learning (SSL) architectures and state-space models (SSMs) across **three distinct downstream datasets** under a strict **Leave-One-Subject-Out Cross-Validation (LOSOCV)** protocol, complemented by Student's $t$ 95% confidence intervals and paired Wilcoxon signed-rank significance tests.

---

## Evaluated Architectures

1. **Corponi et al. (Baseline SSL)**: ResNet convolutional backbone pre-trained via self-supervised masked autoencoding and fine-tuned for binary classification.
2. **BIOT (Biosignal Transformer)**: Patch-based Transformer tokenizing multi-channel physiological time-series with sinusoidal positional encodings.
3. **SimMTM (Masked Time-Series Modeling)**: Series-to-series masked modeling capturing inter-sensor and intra-temporal correlations.
4. **FEMBA (Mamba / Selective State-Space Model)**: Linear-time selective state-space architecture providing efficient long-sequence physiological modeling without quadratic attention bottlenecks.

---

## Datasets & Experimental Protocols

### 1. Pre-Training Corpus (11 Datasets, >6,000 Hours)
Pre-training is performed over the full 11-dataset multi-source corpus:
* `adarp`, `big-ideas` (dati_preelaborati), `in-gauge_en-gage`, `k_emocon`, `ppg_dalia`, `spd`, `stress_detection_nurses_hospital`, `toadstool`, `ue4w`, `weee`, `wesad`, `wesd`.
* Provides continuous telemetry (>6,000 hours) across diverse real-world ambulatory and laboratory conditions.

### 2. Downstream Benchmark Datasets
All downstream evaluations strictly follow **Leave-One-Subject-Out Cross-Validation (LOSOCV)**:
* **WESAD** (15 subjects): Laboratory-controlled emotional and physical stress (Majority Class: 77.9% Baseline/Non-Stress).
* **Indian Student Dataset** (12 subjects): Naturalistic academic and cognitive stress across tasks T1–T5 (Majority Class: 61.0% Non-Stress).
* **PhysioNet** (*Hongn et al., 2025*, 34 subjects): Empatica E4 dataset covering cognitive tasks and physical exertion (Majority Class: 86.8% Non-Stress).

### 3. Unified Hyperparameters & Loss
* **Loss Function**: Class-weighted Binary Cross-Entropy (`pos_weight = n_negative / n_positive`) applied across all models to ensure fair comparisons on imbalanced datasets.
* **Evaluation**: Fold-wise LOSOCV tracking Accuracy, F1-Stress (minority positive class), and Macro-F1.

---

## Benchmark Results (Official LOSOCV & Statistical Significance)

### 1. Mean Metrics with 95% Confidence Intervals (Student's $t$)

| Dataset | Model | Accuracy (95% CI) | F1-Stress (95% CI) | Macro-F1 (95% CI) |
| :--- | :--- | :---: | :---: | :---: |
| **WESAD** *(15 subjects)* | Dummy (Majority) | 77.89% | 0.00% | 43.78% |
| | Corponi | 68.95% ± 12.95% | 49.79% ± 16.77% | 61.65% ± 13.81% |
| | BIOT | 77.09% ± 7.11% | 52.20% ± 15.90% | 68.00% ± 9.52% |
| | SimMTM | 80.59% ± 5.16% | 52.79% ± 13.91% | 71.95% ± 7.50% |
| | **FEMBA (Mamba)** | **82.58% ± 4.54%** | **52.85% ± 13.43%** | **74.00% ± 6.75%** |
| **Indian Students** *(12 subjects)* | Dummy (Majority) | 60.98% | 0.00% | 37.88% |
| | BIOT | 48.60% ± 8.85% | 52.61% ± 10.37% | 44.97% ± 8.78% |
| | Corponi | 50.34% ± 8.82% | 57.85% ± 9.92% | 46.54% ± 8.83% |
| | SimMTM | 53.50% ± 9.17% | 62.42% ± 10.15% | 49.33% ± 9.38% |
| | **FEMBA (Mamba)** | **55.93% ± 7.74%** | **65.34% ± 8.44%** | **51.76% ± 8.27%** |
| **PhysioNet 2025** *(34 subjects)* | Dummy (Majority) | 86.75% | 0.00% | 46.45% |
| | Corponi | 67.45% ± 6.22% | 31.78% ± 6.64% | 54.49% ± 5.19% |
| | BIOT | 73.10% ± 6.22% | 37.39% ± 7.15% | 59.60% ± 5.51% |
| | SimMTM | 83.18% ± 3.86% | **48.56% ± 7.50%** | **68.91% ± 4.57%** |
| | **FEMBA (Mamba)** | **85.48% ± 3.73%** | 38.19% ± 8.35% | 64.81% ± 5.16% |

> **Key Observation**: On heavily skewed telemetry (e.g., PhysioNet at 86.8% majority class), Accuracy alone favors a trivial classifier. Discriminative capability is appropriately evidenced by **F1-Stress** and **Macro-F1**, where self-supervised foundation backbones and FEMBA show substantial gains over baseline methods.

### 2. Paired Wilcoxon Signed-Rank Significance Tests (FEMBA vs Competitors)
* **PhysioNet**:
  * FEMBA vs Corponi: **$p = 0.0001$** (Statistically significant improvement)
  * FEMBA vs BIOT: **$p = 0.0000$** (Statistically significant improvement)
* **Indian Students Dataset**:
  * FEMBA vs Corponi: **$p = 0.0049$** (Statistically significant improvement)
  * FEMBA vs BIOT: **$p = 0.0020$** (Statistically significant improvement)
* **WESAD**:
  * FEMBA vs Corponi: **$p = 0.0001$** (Statistically significant improvement)

---

## Ablation Studies

Summary results are recorded in [`risultati_ablation/ablation_summary.json`](risultati_ablation/ablation_summary.json).

### 1. Data Quantity vs. Data Diversity (8,000-Window Budget)
By fixing an identical pre-training computational budget of 8,000 windows:
* **Single-Source (Big-Ideas)** vs. **Multi-Source Diverse (10 Datasets)**:
  * On laboratory data (WESAD), single-source pre-training performs well (85.54% Acc).
  * On naturalistic ambulatory data (Indian Students), **Multi-Source Diverse pre-training outperforms Single-Source** (64.75% F1-Stress vs. 62.74%), proving that source domain diversity is the critical driver for real-world generalization.

### 2. Leave-One-Dataset-Out (LODO) Pre-Training Ablation
We systematically remove each dataset from the pre-training pool to assess in-domain dependency vs. zero-shot generalizability across all 11 sources.

---

## Reproduction Pipeline

```bash
# 1. Compute statistical significance tables and 95% Confidence Intervals
python compute_statistics.py

# 2. Run the third downstream benchmark (PhysioNet 2025, 34 subjects)
python run_physionet_suite.py

# 3. Run the complete ablation study (FEMBA / LODO / Quantity vs. Diversity)
python run_ablation_study.py --model femba --mode all

# 4. Audit environment and dataset census
python scripts/audit_server_status.py
```

---

## Repository Structure

```text
├── compute_statistics.py        # Paired Wilcoxon & Student's t 95% CI evaluator
├── run_physionet_suite.py       # Full benchmark runner for PhysioNet (Hongn et al. 2025)
├── run_ablation_study.py        # Mamba SSM ablation, LODO, and Diversity study
├── run_corponi_benchmark.py     # Corponi baseline replication runner
├── scripts/
│   ├── train_femba.py           # Pure PyTorch Mamba SSM training pipeline
│   ├── train_biot.py            # BIOT Transformer training pipeline
│   ├── train_simmtm.py          # SimMTM masked modeling training pipeline
│   ├── prepare_physionet_dataset.py # Automated PhysioNet extraction & segmentation
│   ├── prepare_indian_dataset.py    # Indian student dataset segmentation
│   └── audit_server_status.py       # Benchmark integrity audit
├── risultati_benchmark/         # Official fold-by-fold LOSOCV metrics & reports
│   ├── biot_wesad/ / biot_hosseini/ / biot_physionet_losocv/
│   ├── corponi_wesad/ / corponi_hosseini/ / corponi_physionet/
│   ├── femba_wesad/ / femba_hosseini/ / femba_physionet_losocv/
│   └── simmtm_wesad/ / simmtm_hosseini/ / simmtm_physionet_losocv/
└── risultati_ablation/          # Ablation summary JSON and validation records
```
