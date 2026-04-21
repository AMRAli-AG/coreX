# Master Project Guide: OmniAnomaly for CoreX

Welcome to the **OmniAnomaly CoreX** repository! This project implements an advanced deep learning framework for multivariate time-series anomaly detection, specifically tailored for the **CoreX** task (monitoring complex systems like Robot Arms).

---

## 🏗️ Architecture & Versions
The codebase is structured into two standalone versions:
1.  **`Version_1_Baseline/`**: The original OmniAnomaly model. Use this as a benchmark.
2.  **`Version_2_Hybrid/`**: The upgraded architecture featuring a **Causal Graph Module** and **Association Discrepancy** for superior detection accuracy.

---

## 🚀 Installation & Environment Setup

To run this project on a new device, follow these steps to recreate the validated environment.

### 1. Create Conda Environment
It is highly recommended to use **Python 3.6** for maximum compatibility with TensorFlow 1.15.
```bash
conda create -n corex_env python=3.6
conda activate corex_env
```

### 2. Install Core Dependencies
Install the required deep learning and data processing libraries:
```bash
pip install -r Version_2_Hybrid/requirements.txt
```
*(This includes TensorFlow 1.15.0, TF-Probability 0.7.0, Seaborn, and PyYAML 5.3.1)*

### 3. GPU Acceleration (Recommended)
If you have an NVIDIA GPU, install the required CUDA Toolkit and cuDNN libraries via Conda to enable hardware acceleration:
```bash
conda install -c anaconda cudatoolkit=10.0 cudnn=7
```

---

## 📂 Repository Structure
```text
/
├── Version_1_Baseline/     # Baseline model, data, and checkpoints
├── Version_2_Hybrid/       # Optimized Hybrid model (Causal Graph + GRU)
├── notebooks/              # Cloud-ready notebooks (Colab/Kaggle)
├── README_GLOBAL.md        # You are reading this!
└── requirements.txt        # Shared dependency list
```

---

## 🏃 Quick Run Commands

### To Evaluate Pre-trained Models (Scoring Only)
If you just want to see the results without training:
```bash
# For Version 1
cd Version_1_Baseline
python main.py --restore_dir model_coreX_v1 --max_epoch 0

# For Version 2
cd Version_2_Hybrid
python main.py --restore_dir model_coreX_v2_optimized --max_epoch 0
```

### To Train from Scratch
```bash
python main.py
```

*Note: All datasets are pre-included in the `data/processed/` folders within each version.*
