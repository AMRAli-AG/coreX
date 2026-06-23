# Kaggle Notebook Guide - OmniAnomaly Version 2 Hybrid

This guide explains how to set up, upload, and run the OmniAnomaly Version 2 Hybrid model on Kaggle using the pre-built notebook **`corex-v2-kaggle.ipynb`**.

---

## 1. What to Upload to Kaggle

You need to upload **two things** to Kaggle as datasets (using the "Add Data" button):

1. **The Codebase ZIP (`corex_v2_code.zip`)**:
   - Run the local script `zip_codebase.py` to package the codebase. This will generate `corex_v2_code.zip` in your local `OMNI` folder.
   - Upload this ZIP file as a Kaggle dataset.

2. **The Dataset File (flexible - upload ONE of the following)**:
   - **Option A (Raw data)**: Upload `all_data.csv`. The preprocessing script will automatically detect it and unpack it.
   - **Option B (Pre-unpacked data)**: Upload `ready_for_ai.csv`. The preprocessing script will detect it, bypass unpacking, and load it directly.

---

## 2. Kaggle Session Settings

Before running any cells, configure these settings in the **right-hand panel** of your Kaggle Notebook:
1. **Internet**: Set to **On** (required to install Conda and Pip packages).
2. **Accelerator**: Set to **GPU T4** (or **GPU T4 x2**).

---

## 3. Cell Explanations

### Cell 1: Extract Codebase & Setup Data
- **What it does**: Automatically detects and extracts the `corex_v2_code.zip` file into the `/kaggle/working/Version_2_Hybrid` directory. It then finds either `all_data.csv` or `ready_for_ai.csv` in your Kaggle datasets and copies it to the appropriate data folder.

### Cell 2: Environment Setup
- **What it does**: Sets up the legacy environment needed for this codebase. It installs `Miniconda`, accepts Conda's TOS, creates a `Python 3.6` environment, installs `TensorFlow 1.15 GPU`, and patches `tfsnippet` internally to fix dimension mismatch issues during execution.

### Cell 3: Verify GPU Availability
- **What it does**: A quick sanity check to ensure that TensorFlow detects and utilizes the Kaggle GPU hardware.

### Cell 4: Configure Epochs
- **What it does**: Contains a python script that edits the `main.py` configuration to set the desired number of training epochs (`EPOCHS` variable). This makes it easy to switch between a quick 1-epoch test and a full production training cycle.

### Cell 5: Run Data Preprocessing
- **What it does**: Executes `data_preprocess.py` inside the Version 2 Hybrid codebase to handle feature engineering, temporal context preparation, causal relationships integration, and data splitting.

### Cell 6: Train the Model
- **What it does**: Triggers `main.py`, kicking off the training for the hybrid temporal-context and causal graph anomaly detection model.

### Cell 7: Generate Evaluation Report & Plots
- **What it does**: Runs `plot_results.py` to visualize the model's performance, losses, and detection scores. 

### Cell 8: Zip and Export Results
- **What it does**: Packages the `results` and `model_coreX_v2_optimized` folders into downloadable `.zip` files. You can find and download these zips from Kaggle's "Output" pane once the run finishes.
