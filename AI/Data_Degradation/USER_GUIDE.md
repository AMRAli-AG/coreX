# Comprehensive User Guide: Data Degradation Framework

Welcome to the **Data Degradation Framework**. This guide is designed to help you quickly understand what this folder contains, how the scripts interact, and exactly how to run the integrated pipeline.

---

## 📂 Folder Anatomy
This folder is structured into two main stages: **Data Augmentation** and **Fault Injection**.

### 1. `main.py` (The Control Center)
This is the only script you need to run directly. It acts as an interactive terminal dashboard. It coordinates the expansion of your data and the injection of faults.

### 2. `/augmentation/` (Stage 1)
- **`expander.py`**: Contains the logic to take your baseline `ready_for_ai.csv` dataset and multiply its size (e.g., 2x, 5x, 10x). It intelligently stitches the time-series data together so that the robotic movements remain physically continuous and valid.

### 3. `/injectors/` (Stage 2)
This directory houses the "Diseases" — the mathematical models that corrupt the healthy data.
- **`base_injector.py`**: The parent template that all faults follow.
- **`/local/backlash.py`**: Simulates gear wear. It causes a positional lag whenever the motor changes direction.
- **`/local/friction.py`**: Simulates lubrication failure. It linearly forces the motor to draw more current and increases the joint temperature.
- **`/local/bearing_wear.py`**: Simulates pitting in the bearings. It introduces high-frequency noise and extreme amplitude spikes (increasing Kurtosis) to the current and velocity.
- **`/propagation/payload_collision.py`**: A system-wide fault. Simulates the robot hitting an object or picking up a heavy payload, forcing *all* joints to draw 50% more current.
- **`/combos/combo.py`**: Allows you to stack multiple faults (e.g., Backlash + Bearing Wear) simultaneously on the same joint.

### 4. `/utils/`
- **`data_manager.py`**: Automatically reads the columns of your CSV and figures out which ones are `current`, `position (q)`, `velocity (qd)`, etc. It also loads large datasets in chunks so your computer doesn't run out of memory.
- **`labeler.py`**: Automatically adds a `fault_label` column and a continuous `severity_score` (from 0.0 to 1.0) column to your data, making it perfectly prepped for Machine Learning.

### 5. Documentation Files
- **`Technical_Review.md`**: Read this if you want to understand the exact mathematical formulas used to simulate the physics of the faults.
- **`requirements.txt`**: The list of Python packages needed to run this framework.

---

## 🚀 How to Run the Pipeline

### Step 1: Preparation
Ensure your healthy robotic dataset is named **`ready_for_ai.csv`** and is placed in this main folder.
Make sure you have installed the requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Engine
Open your terminal in this folder and type:
```bash
python main.py
```

### Step 3: Follow the Interactive Prompts
The terminal will guide you through the process:

1. **Analysis**: The script will print out how many rows and columns are in your `ready_for_ai.csv`.
2. **Expansion**: It will ask: *"Do you want to expand the dataset size? (Yes/No)"*
   - Type `Yes` and enter a multiplier (like `2` or `5`) if you want more training data.
   - Type `No` to just use the original file.
3. **Select a Disease**: It will present a menu. Type the abbreviation for the fault you want:
   - `BL` (Backlash)
   - `LF` (Lubrication/Friction)
   - `BW` (Bearing Wear)
   - `SYS` (System/Payload)
   - `COMBO` (Multiple)
4. **Target Location & Time**: 
   - Enter the **Joint Index** you want to corrupt (e.g., `0` for the first joint).
   - Enter the **Starting Row** (when the fault should begin).
   - Enter the **Duration** (how many rows the fault should last before reaching maximum severity).

### Step 4: The Result
Once the script finishes:
1. A new file called **`labeled_robot_data.csv`** will be generated. This is your final ML-ready dataset.
2. A summary report will be printed in the terminal.
3. A beautiful **Matplotlib graph** will automatically open, showing you a visual comparison of the healthy signal vs. your newly injected faulty signal, color-coded by severity!
