# Kaggle Notebook Guide - OmniAnomaly Version 1 Baseline

This guide explains how to set up, upload, and run the OmniAnomaly Version 1 Baseline model on Kaggle. You can either import the pre-built notebook **`corex-v1-baseline-kaggle.ipynb`** directly into Kaggle or copy the cells.

---

## 1. What to Upload to Kaggle

You need to upload the following as datasets (using the "Add Data" button):

1. **The Codebase ZIP (`Version_1_Baseline` folder compressed or as a dataset)**:
   - Ensure the folder `Version_1_Baseline` is uploaded.

2. **The Dataset File**:
   - Upload `all_data.csv`. The script will automatically copy it to the correct project structure inside Kaggle.

---

## 2. Kaggle Session Settings

Before running any cells, configure these settings in the **right-hand panel** of your Kaggle Notebook:
1. **Internet**: Set to **On** (required to install Conda and Pip packages).
2. **Accelerator**: Set to **GPU T4** (or **GPU T4 x2**).

---

## 3. Cell Explanations

### Cell 1: Copy Project files & Setup Data
- **What it does**: Locates the `Version_1_Baseline` codebase folder and `all_data.csv` dataset you uploaded. It then prepares a clean `project/` directory inside `/kaggle/working` and copies these assets so the environment has everything in place.

### Cell 2: Environment Setup (Python 3.6 + TF 1.15)
- **What it does**: Kaggle's default environment uses a newer Python version incompatible with TF 1.15. This cell installs `Miniconda`, sets up a Python 3.6 environment (`corex_env`), installs TensorFlow 1.15 GPU, CUDA requirements, and patches `tfsnippet` from source.

### Cell 3: Verify GPU
- **What it does**: Quickly runs a python check to ensure TensorFlow is utilizing the GPU correctly.

### Cell 4: Install Extra Dependencies
- **What it does**: Installs extra packages like `tensorflow-probability`, `tqdm`, `PyYAML`, and spreadsheet libraries required for the project.

### Cell 5: Patch main.py & plot_results.py Config
- **What it does**: Patches the python scripts (`main.py` and `plot_results.py`) dynamically to train for 25 epochs (by default), adjust the batch size, ensure fresh training without trying to restore old models, and correctly map output paths for result visualization.

### Cell 6: Unpack & Preprocess Data
- **What it does**: Runs `data_preprocess.py` which un-flattens the sensor arrays, engineers features, structures them properly for OmniAnomaly, and introduces synthetic anomalies.

### Cell 7: Train the Model
- **What it does**: Executes `main.py` to train the Multivariate Time-Series Anomaly Detection model (VAE + Normalizing Flows) for the configured number of epochs.

### Cell 8: Generate Baseline Performance Plot
- **What it does**: Executes `plot_results.py` to generate evaluation plots and anomaly detection result graphs.

### Cell 9: Export Results & Checkpoints
- **What it does**: Zips the generated `results/` and `model_coreX_v1/` output directories into downloadable files (`coreX_v1_baseline_results.zip` and `coreX_v1_baseline_model.zip`), so you can easily download them from the Kaggle Output tab.
