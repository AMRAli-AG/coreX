# Kaggle Notebook Guide - OmniAnomaly Version 2 Hybrid

This guide explains how to set up, upload, and run the OmniAnomaly Version 2 Hybrid model on Kaggle. You can either import the pre-built notebook **`notebooks/corex-v2-kaggle.ipynb`** directly into Kaggle, or create a new notebook and copy-paste the cells below.

---

## 1. What to Upload to Kaggle

You need to upload **two things** to Kaggle as datasets (using the "Add Data" button):

1. **The Codebase ZIP (`corex_v2_code.zip`)**:
   - Run the local script `zip_codebase.py` to package the codebase. This will generate `corex_v2_code.zip` in your local `OMNI` folder.
   - Upload this ZIP file as a Kaggle dataset.

2. **The Dataset File (flexible - upload ONE of the following)**:
   - **Option A (Raw data)**: Upload `all_data.csv`. The preprocessing script will automatically detect it and unpack it.
   - **Option B (Pre-unpacked data)**: Upload `ready_for_ai.csv`. The preprocessing script will detect it, bypass unpacking, and load it directly.
   - **Note**: The notebook is fully automated and will auto-detect whichever file you choose to upload.

---

## 2. Kaggle Session Settings

Before running any cells, configure these settings in the **right-hand panel** of your Kaggle Notebook:
1. **Internet**: Set to **On** (required to install Conda and Pip packages).
2. **Accelerator**: Set to **GPU T4** (or **GPU T4 x2**).

---

## 3. How to Run a 1-Epoch Sanity Check

To verify that the environment, data loading, preprocessing, and model layers are working correctly without waiting hours:
1. Go to **Cell 4** (Configure Epochs).
2. Set `EPOCHS = 1`.
3. Run all cells. The model will run 1 training epoch and complete successfully, generating validation reports and output files.
4. Once you confirm it works, change `EPOCHS = 50` (or `100`) in **Cell 4** and run the cells again for full training.

---

## 4. Notebook Cell Code

### Cell 1: Extract Codebase & Setup Data (Python)
```python
import os
import shutil
import zipfile
import glob

WORKING_DIR = "/kaggle/working/"
code_dir = os.path.join(WORKING_DIR, 'Version_2_Hybrid')
data_dir = os.path.join(code_dir, 'data', 'RobotArm')

print("=== 1. Detecting and Extracting Codebase ===")
zip_path = None
# Search for corex_v2_code.zip
zip_matches = glob.glob("/kaggle/input/**/corex_v2_code.zip", recursive=True)
if zip_matches:
    zip_path = zip_matches[0]
    print(f"🔍 Found codebase zip at: {zip_path}")
else:
    # Find any zip file if not named corex_v2_code.zip
    all_zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if all_zips:
        zip_path = all_zips[0]
        print(f"🔍 Found ZIP file: {zip_path}, assuming it is the codebase.")

if zip_path:
    print(f"Extracting {zip_path} to {WORKING_DIR}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(WORKING_DIR)
    print("✅ Codebase extracted successfully to /kaggle/working/Version_2_Hybrid")
else:
    print("⚠️ ZIP file not found. Searching for pre-extracted codebase folder...")
    found_dir = None
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'data_preprocess.py' in files:
            found_dir = root
            break
    if found_dir:
        print(f"🔍 Found extracted codebase directory at: {found_dir}")
        if os.path.exists(code_dir):
            shutil.rmtree(code_dir)
        shutil.copytree(found_dir, code_dir)
        print("✅ Codebase copied successfully to /kaggle/working/Version_2_Hybrid")
    else:
        print("❌ ERROR: Could not find codebase zip or extracted codebase folder containing 'data_preprocess.py' in /kaggle/input!")

print("\n=== 2. Detecting and Copying Data File ===")
os.makedirs(data_dir, exist_ok=True)
csv_matches = glob.glob("/kaggle/input/**/all_data.csv", recursive=True)
ready_matches = glob.glob("/kaggle/input/**/ready_for_ai.csv", recursive=True)

if csv_matches:
    src_csv = csv_matches[0]
    dest_csv = os.path.join(data_dir, 'all_data.csv')
    shutil.copy(src_csv, dest_csv)
    print(f"✅ Found raw dataset (all_data.csv). Copied to: {dest_csv}")
elif ready_matches:
    src_ready = ready_matches[0]
    dest_ready = os.path.join(data_dir, 'ready_for_ai.csv')
    shutil.copy(src_ready, dest_ready)
    print(f"✅ Found pre-unpacked dataset (ready_for_ai.csv). Copied to: {dest_ready}")
else:
    # Dynamic fallback
    all_csvs = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    external_csvs = [c for c in all_csvs if 'Version_2_Hybrid' not in c]
    if external_csvs:
        src_csv = external_csvs[0]
        dest_csv = os.path.join(data_dir, 'all_data.csv')
        shutil.copy(src_csv, dest_csv)
        print(f"✅ Found alternative CSV file ({src_csv}). Copied to: {dest_csv}")
    else:
        print("❌ ERROR: Could not find all_data.csv or ready_for_ai.csv in /kaggle/input!")
```

### Cell 2: Environment Setup (Bash)
```bash
%%bash
set -e

echo "=== 1. Checking conda ==="
if ! command -v /opt/conda/bin/conda &> /dev/null; then
    echo "=== Installing Miniconda ==="
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda -f 2>/dev/null
fi
export PATH=/opt/conda/bin:$PATH

echo "=== 2. Configuring Conda channel options & accepting ToS ==="
conda config --set always_yes true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "=== 3. Creating corex_env with Python 3.6 ==="
conda env remove -n corex_env 2>/dev/null || true
conda create -n corex_env python=3.6 -y -q

echo "=== 4. Installing TensorFlow 1.15 GPU and CUDA Stack (takes ~3 mins) ==="
conda install -n corex_env -c defaults tensorflow-gpu=1.15 cudatoolkit=10.0 cudnn=7 numpy=1.16 scipy pandas scikit-learn matplotlib -y -q

echo "=== 5. Installing Specialized Dependencies ==="
/opt/conda/envs/corex_env/bin/pip install -q \
    git+https://github.com/haowen-xu/tfsnippet.git@63adaf04d2ffff8dec299623627d55d4bacac598 \
    git+https://github.com/thu-ml/zhusuan.git \
    seaborn \
    imageio \
    tensorflow-probability==0.8.0 \
    tqdm==4.28.1 \
    fs==2.3.0 \
    click==7.0 \
    PyYAML==5.4.1 \
    xlrd==2.0.2 \
    openpyxl==3.1.3 \
    frozendict>=2.0.6 \
    idx2numpy>=1.2.3 \
    natsort>=7.1.1 \
    semver>=2.7.9 \
    lazy-object-proxy>=1.4.3

echo "=== 6. Patching tfsnippet flow.py ==="
/opt/conda/envs/corex_env/bin/python -c "
import os
path = '/opt/conda/envs/corex_env/lib/python3.6/site-packages/tfsnippet/distributions/flow.py'
if os.path.exists(path):
    with open(path, 'r') as f:
        c = f.read()
    c = c.replace('group_ndims=ndims_diff,', 'group_ndims=self.flow.x_value_ndims,')
    c = c.replace('log_px = self._distribution.log_prob(x, group_ndims=ndims_diff)', 'log_px = self._distribution.log_prob(x, group_ndims=self.flow.x_value_ndims)')
    with open(path, 'w') as f:
        f.write(c)
    print('✅ Patched tfsnippet flow.py successfully!')
else:
    print('❌ Could not find flow.py to patch!')
"

echo "✅ Environment configured successfully!"
```

### Cell 3: Verify GPU Availability (Bash)
```bash
%%bash
export PATH=/opt/conda/envs/corex_env/bin:$PATH
/opt/conda/envs/corex_env/bin/python -c "
import tensorflow as tf
print('TensorFlow Version:', tf.__version__)
print('GPU Available:', tf.test.is_gpu_available())
print('GPU Name:', tf.test.gpu_device_name())
"
```

### Cell 4: Configure Epochs (Python)
```python
# ⚡ CONFIGURE TRAINING EPOCHS HERE
# Set to 1 for a quick sanity-check test run.
# Set to 50 or 100 for full production training.
EPOCHS = 1

import os
import re

project_dir = "/kaggle/working/Version_2_Hybrid"
if os.path.exists(project_dir):
    os.chdir(project_dir)
    main_path = "main.py"
    if os.path.exists(main_path):
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Update max_epoch
        content = re.sub(r'max_epoch\s*=\s*\d+', f'max_epoch = {EPOCHS}', content)
        # Ensure fresh training (no restore_dir)
        content = re.sub(r"restore_dir\s*=\s*['\"](.*?)['\"]", 'restore_dir = None', content)
        content = re.sub(r"restore_dir\s*=\s*None", 'restore_dir = None', content)
        
        # Force optimal batch_size for memory management
        content = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 64', content)
        
        with open(main_path, 'w') as f:
            f.write(content)
        print(f"✅ Patched main.py to train for {EPOCHS} epochs.")
    else:
        print("❌ ERROR: main.py not found!")
else:
    print("❌ ERROR: Project directory not found!")
```

### Cell 5: Run Data Preprocessing (Bash)
```bash
%%bash
cd /kaggle/working/Version_2_Hybrid
export PATH=/opt/conda/envs/corex_env/bin:$PATH
export MPLBACKEND=Agg
export PYTHONPATH=.

echo "=== Starting Preprocessing & Feature Engineering ==="
/opt/conda/envs/corex_env/bin/python -u data_preprocess.py
```

### Cell 6: Train the Model (Bash)
```bash
%%bash
cd /kaggle/working/Version_2_Hybrid
export PATH=/opt/conda/envs/corex_env/bin:$PATH
export MPLBACKEND=Agg
export PYTHONPATH=.

echo "=== Starting Model Training (Version 2 Hybrid) ==="
/opt/conda/envs/corex_env/bin/python -u main.py
```

### Cell 7: Generate Evaluation Report & Plots (Bash)
```bash
%%bash
cd /kaggle/working/Version_2_Hybrid
export PATH=/opt/conda/envs/corex_env/bin:$PATH
export MPLBACKEND=Agg
export PYTHONPATH=.

echo "=== Generating Evaluation Report & Plots ==="
if [ -f "plot_results.py" ]; then
    /opt/conda/envs/corex_env/bin/python -u plot_results.py
else
    echo "⚠️ plot_results.py not found, skipping."
fi
```

### Cell 8: Zip and Export Results (Python)
```python
import os
import glob
import zipfile

os.chdir("/kaggle/working/Version_2_Hybrid")

def make_zip(name, patterns):
    zip_path = f"/kaggle/working/{name}"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in patterns:
            for f in glob.glob(p, recursive=True):
                z.write(f, os.path.relpath(f, '.'))
    print(f"✅ Successfully created {name} at {zip_path}")

if os.path.exists('results'):
    make_zip('corex_v2_results.zip', ['results/**/*'])
else:
    print("⚠️ Warning: results folder not found to zip!")

if os.path.exists('model_coreX_v2_optimized'):
    make_zip('corex_v2_model.zip', ['model_coreX_v2_optimized/**/*'])
else:
    print("⚠️ Warning: checkpoints folder not found to zip!")

print("\n🎉 Complete! Download corex_v2_results.zip and corex_v2_model.zip from Kaggle's output folder.")
```
