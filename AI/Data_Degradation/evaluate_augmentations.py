# -*- coding: utf-8 -*-
"""
Robotic Telemetry Augmentation Evaluator & Benchmark Suite.
Author: Antigravity AI Partner
Date: 2026-05-21

Automates:
1. Dynamic Scanning of the 'augmentation' folder for modular techniques.
2. Sequence windowing, scaling, and training data partitioning.
3. Automated Evaluation (TSTR - Train on Synthetic, Test on Real).
4. Generation of 'augmentation_accuracy.csv' and professional console diagnostics.
"""

import os
import sys
import time
import importlib.util
import warnings
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Suppress annoying runtime warnings
warnings.filterwarnings('ignore')

# Configurable constants for high-fidelity evaluation
SEQ_LEN = 24  # Sequence length/window size
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 30  # Optimized for high speed and accurate convergence
MAX_DEPTH = 15     # Prevents overfitting on flattened sequence vectors


def print_banner():
    """Prints a beautiful ASCII banner representing the diagnostic engine."""
    print("\n" + "=" * 85)
    print("   AUTOMATED DATA AUGMENTATION EVALUATION & TSTR DIAGNOSTIC SUITE   ".center(85))
    print("=" * 85)
    print("   Physically Informed Diagnostics | Dynamic scanning | Industry 4.0 Standard   ".center(85))
    print("=" * 85 + "\n")


def scan_augmentation_techniques(folder: str = "augmentation") -> List[Tuple[str, str, Any]]:
    """
    Dynamically scans the given folder for Python modules representing augmentation techniques.
    Loads them dynamically, ignoring internal helpers and the orchestrating expander.
    
    Args:
        folder (str): Augmentation directory to scan.
        
    Returns:
        List[Tuple[str, str, Any]]: List of tuples containing (technique_name, file_path, module).
    """
    techniques = []
    if not os.path.exists(folder):
        print(f"[!] Warning: Folder '{folder}' not found. Creating empty folder.")
        os.makedirs(folder, exist_ok=True)
        return techniques

    print(f"[*] Scanning folder '{folder}' for telemetry augmentation techniques...")
    
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".py") and filename != "expander.py" and not filename.startswith("__"):
            filepath = os.path.join(folder, filename)
            module_name = filename[:-3]
            
            try:
                # Dynamically load the python module
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Fetch human-readable name or fall back to module name
                tech_name = getattr(module, "TECHNIQUE_NAME", module_name.replace("_", " ").title())
                
                # Validate that the technique implements at least one standard interface
                has_seq_interface = hasattr(module, "augment_sequences")
                has_df_interface = hasattr(module, "augment_dataframe")
                
                if has_seq_interface or has_df_interface:
                    techniques.append((tech_name, filepath, module))
                    print(f"    [OK] Detected '{tech_name}' ({filename})")
                    if has_seq_interface:
                        print("         -> Interface: augment_sequences(X, y)")
                    if has_df_interface:
                        print("         -> Interface: augment_dataframe(df, target_col)")
                else:
                    print(f"    [SKIP] Module '{filename}' lacks a valid augment function. Skipped.")
                    
            except Exception as e:
                print(f"    [ERROR] Failed to load '{filename}': {e}")
                
    print(f"[#] Total techniques detected: {len(techniques)}\n")
    return techniques


def load_and_preprocess_base_data(filepath: str = "ready_for_ai.csv") -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads base telemetry dataset and extracts features and target columns."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Critical base dataset file '{filepath}' is missing!")
        
    print(f"[*] Loading baseline healthy telemetry dataset: '{filepath}'...")
    df = pd.read_csv(filepath)
    print(f"    -> Data shape: {df.shape[0]} samples, {df.shape[1]} channels")
    
    # Identify target column (robot mode)
    if 'robot_mode' in df.columns:
        target_col = 'robot_mode'
    elif 'Target' in df.columns:
        target_col = 'Target'
    else:
        target_col = df.columns[-1]  # Default to last column
        
    features = [c for c in df.columns if c != target_col]
    print(f"    -> Target column identified: '{target_col}'")
    print(f"    -> Numerical features count: {len(features)}")
    
    return df, target_col, features


def create_windowed_sequences(data: np.ndarray, target: np.ndarray, window_size: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Converts 2D flat sequence matrix into 3D windowed structures for temporal modeling."""
    X_windows = []
    y_windows = []
    
    # Efficient rolling windows
    for i in range(len(data) - window_size + 1):
        X_windows.append(data[i:i+window_size])
        y_windows.append(target[i+window_size-1])  # Keep target label of the last timestamp
        
    return np.array(X_windows), np.array(y_windows)


def run_evaluation_pipeline(
    df: pd.DataFrame, 
    target_col: str, 
    features: List[str], 
    techniques: List[Tuple[str, str, Any]]
) -> pd.DataFrame:
    """
    Main evaluation pipeline. Standardizes data, creates base splits, 
    applies detected augmentations, trains classifiers, and records metrics.
    """
    # 1. Standardize base features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)
    y = df[target_col].values
    
    # 2. Form baseline temporal sequences
    print("[*] Performing windowing segmentation (Window Size = 24)...")
    X_seq, y_seq = create_windowed_sequences(X_scaled, y, window_size=SEQ_LEN)
    print(f"    -> Sequential tensor dimensions: {X_seq.shape}")
    
    # 3. Train-Test Split (TSTR requires testing strictly on real data)
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Optimization: Downsample training subset to speed up random forest training
    MAX_TRAIN_SAMPLES = 4000
    if len(X_train_real) > MAX_TRAIN_SAMPLES:
        print(f"    [~] Downsampling training set from {len(X_train_real)} to {MAX_TRAIN_SAMPLES} sequences for high-speed fitting...")
        np.random.seed(RANDOM_STATE)
        idx_train = np.random.choice(len(X_train_real), MAX_TRAIN_SAMPLES, replace=False)
        X_train_real_sub = X_train_real[idx_train]
        y_train_real_sub = y_train_real[idx_train]
    else:
        X_train_real_sub = X_train_real
        y_train_real_sub = y_train_real
        
    results = []
    
    # ----------------------------------------------------
    # BASELINE: Train on Real, Test on Real
    # ----------------------------------------------------
    print("\n[*] Establishing Baseline Performance (Train on Real -> Test on Real)...")
    start_t = time.time()
    
    # Flatten sequential features for traditional machine learning models (Random Forest)
    X_train_real_flat = X_train_real_sub.reshape(X_train_real_sub.shape[0], -1)
    X_test_real_flat = X_test_real.reshape(X_test_real.shape[0], -1)
    
    clf_base = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1)
    clf_base.fit(X_train_real_flat, y_train_real_sub)
    
    preds_base = clf_base.predict(X_test_real_flat)
    latency_base = time.time() - start_t
    
    acc_base = accuracy_score(y_test_real, preds_base)
    f1_base = f1_score(y_test_real, preds_base, average='weighted')
    prec_base = precision_score(y_test_real, preds_base, average='weighted', zero_division=0)
    rec_base = recall_score(y_test_real, preds_base, average='weighted', zero_division=0)
    
    print(f"    [BASELINE] Accuracy: {acc_base:.4f} | F1-Score: {f1_base:.4f} ({latency_base:.2f}s)")
    
    results.append({
        'Technique': 'Baseline (Real Data Only)',
        'TSTR Accuracy': acc_base,
        'TSTR F1-Score': f1_base,
        'TSTR Precision': prec_base,
        'TSTR Recall': rec_base,
        'Augmentation Time (s)': 0.0,
        'Training Time (s)': latency_base,
        'Samples Generated': 0,
        'Status': 'Baseline Standard'
    })
    
    # ----------------------------------------------------
    # DYNAMIC TECHNIQUES BENCHMARKING
    # ----------------------------------------------------
    for tech_name, file_path, module in techniques:
        print(f"\n[*] Evaluating Technique: '{tech_name}'...")
        
        try:
            start_aug = time.time()
            
            # Scenario A: Module implements dataframe-level augmentation (much faster, vectorized!)
            if hasattr(module, "augment_dataframe"):
                print(f"    [+] Applying dataframe augmentation on raw split...")
                # Split raw data first to ensure no test data leakage
                df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                
                # Downsample flat df_train before dataframe augmentation to save time
                MAX_DF_TRAIN_SAMPLES = 4000
                if len(df_train) > MAX_DF_TRAIN_SAMPLES:
                    df_train_sub = df_train.sample(n=MAX_DF_TRAIN_SAMPLES, random_state=RANDOM_STATE)
                else:
                    df_train_sub = df_train
                    
                df_train_aug = module.augment_dataframe(df_train_sub, target_col)
                aug_duration = time.time() - start_aug
                
                # Transform augmented df to scaled sequential data
                X_train_aug_scaled = scaler.transform(df_train_aug[features].values)
                y_train_aug_raw = df_train_aug[target_col].values
                X_train_aug, y_train_aug = create_windowed_sequences(
                    X_train_aug_scaled, y_train_aug_raw, window_size=SEQ_LEN
                )
                samples_gen = len(X_train_aug) - len(df_train_sub)
                
            # Scenario B: Module implements sequence-level augmentation
            elif hasattr(module, "augment_sequences"):
                print(f"    [+] Applying sequence augmentation on 3D training arrays...")
                X_train_aug, y_train_aug = module.augment_sequences(X_train_real_sub, y_train_real_sub)
                aug_duration = time.time() - start_aug
                samples_gen = len(X_train_aug) - len(X_train_real_sub)
            
            else:
                raise NotImplementedError("Technique does not implement a valid interface.")
            
            print(f"    [+] Successfully synthesized {samples_gen} new training samples.")
            
            # Flatten augmented data for training
            X_train_aug_flat = X_train_aug.reshape(X_train_aug.shape[0], -1)
            
            # Train model under same parameters
            print(f"    [+] Training Random Forest Classifier on augmented training set...")
            start_train = time.time()
            clf_tech = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )
            clf_tech.fit(X_train_aug_flat, y_train_aug)
            train_duration = time.time() - start_train
            
            # Evaluate strictly on real test data (TSTR)
            preds_tech = clf_tech.predict(X_test_real_flat)
            
            acc_tech = accuracy_score(y_test_real, preds_tech)
            f1_tech = f1_score(y_test_real, preds_tech, average='weighted')
            prec_tech = precision_score(y_test_real, preds_tech, average='weighted', zero_division=0)
            rec_tech = recall_score(y_test_real, preds_tech, average='weighted', zero_division=0)
            
            print(f"    [OK] TSTR Accuracy: {acc_tech:.4f} | F1-Score: {f1_tech:.4f}")
            
            results.append({
                'Technique': tech_name,
                'TSTR Accuracy': acc_tech,
                'TSTR F1-Score': f1_tech,
                'TSTR Precision': prec_tech,
                'TSTR Recall': rec_tech,
                'Augmentation Time (s)': aug_duration,
                'Training Time (s)': train_duration,
                'Samples Generated': samples_gen,
                'Status': 'Success'
            })
            
        except Exception as e:
            print(f"    [FAILED] Error evaluating '{tech_name}': {e}")
            results.append({
                'Technique': tech_name,
                'TSTR Accuracy': 0.0,
                'TSTR F1-Score': 0.0,
                'TSTR Precision': 0.0,
                'TSTR Recall': 0.0,
                'Augmentation Time (s)': 0.0,
                'Training Time (s)': 0.0,
                'Samples Generated': 0,
                'Status': f"Failed: {str(e)}"
            })
            
    return pd.DataFrame(results)


def export_and_display_results(results_df: pd.DataFrame, output_path: str = "augmentation_accuracy.csv"):
    """Saves results to a CSV file and prints a high-fidelity diagnostic console report."""
    # Export to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\n[#] Technical report saved successfully to: '{output_path}'\n")
    
    # Display gorgeous tabular console report
    print("=" * 115)
    print(f"   AUGMENTATION ACCURACY EXPERIMENT RESULTS TABLE ({len(results_df) - 1} Techniques Tested)   ".center(115))
    print("=" * 115)
    print(
        f"{'Technique':<30} | {'TSTR Acc':<10} | {'TSTR F1':<10} | {'TSTR Prec':<10} | "
        f"{'Gen Time (s)':<12} | {'Train Time (s)':<14} | {'Synthetic Samples':<18}"
    )
    print("-" * 115)
    
    for _, row in results_df.iterrows():
        # Highlight Baseline
        name = row['Technique']
        if name.startswith('Baseline'):
            name = f"==> {name}"
            
        print(
            f"{name:<30} | {row['TSTR Accuracy']:<10.4f} | {row['TSTR F1-Score']:<10.4f} | "
            f"{row['TSTR Precision']:<10.4f} | {row['Augmentation Time (s)']:<12.2f} | "
            f"{row['Training Time (s)']:<14.2f} | {int(row['Samples Generated']):<18}"
        )
    print("=" * 115 + "\n")


def main():
    print_banner()
    
    try:
        # Step 1: Detect base dataset
        base_file = "ready_for_ai.csv"
        df, target_col, features = load_and_preprocess_base_data(base_file)
        
        # Step 2: Dynamically scan the augmentation/ folder
        techniques = scan_augmentation_techniques("augmentation")
        
        if not techniques:
            print("[!] Warning: No valid modular techniques detected in the 'augmentation' folder.")
            print("[!] Please check that technique scripts exist under 'augmentation/' and implement valid interfaces.")
            sys.exit(1)
            
        # Step 3: Run all evaluations
        results_df = run_evaluation_pipeline(df, target_col, features, techniques)
        
        # Step 4: Export report and display
        export_and_display_results(results_df, "augmentation_accuracy.csv")
        
        # Step 5: Interactive Selection & Generation
        print("\n" + "=" * 85)
        print("                 STAGE 1 INTERACTIVE SELECTION & EXPORT                  ")
        print("=" * 85)
        
        selectable = [r for _, r in results_df.iterrows() if not r['Technique'].startswith('Baseline')]
        
        print("\nAvailable Augmentation Techniques:")
        for idx, r in enumerate(selectable, 1):
            status = r.get('Status', 'Success')
            status_str = f" [Status: {status}]" if "Failed" in str(status) else ""
            print(f"  [{idx}] {r['Technique']:<25} (TSTR Accuracy: {r['TSTR Accuracy']:.4f}){status_str}")
        print("  [0] Skip / Use Baseline Healthy Data (No Augmentation)")
        
        while True:
            try:
                choice_str = input(f"\n[>] Select technique to apply (0-{len(selectable)}) [0]: ").strip()
                if not choice_str:
                    choice = 0
                    break
                choice = int(choice_str)
                if 0 <= choice <= len(selectable):
                    break
                print("[!] Invalid selection. Please choose a valid index.")
            except ValueError:
                print("[!] Please enter a valid integer.")
                
        import pickle
        if choice == 0:
            print("\n[*] Skipping augmentation. Baseline healthy telemetry dataset will be used.")
            export_data = {
                'technique': 'Baseline (Real Data Only)',
                'ratio': 1,
                'dataframe': df
            }
            with open("selected_augmentation.pkl", "wb") as f:
                pickle.dump(export_data, f)
            print("[OK] Reference baseline exported to 'selected_augmentation.pkl'.\n")
        else:
            selected_row = selectable[choice - 1]
            tech_name = selected_row['Technique']
            print(f"\n[+] Selected Technique: '{tech_name}'")
            
            while True:
                try:
                    ratio_str = input("[>] Enter desired Expansion Ratio (multiplier, e.g., 2, 3, 5) [2]: ").strip()
                    if not ratio_str:
                        ratio = 2
                        break
                    ratio = int(ratio_str)
                    if ratio >= 1:
                        break
                    print("[!] Ratio must be a positive integer >= 1.")
                except ValueError:
                    print("[!] Please enter a valid integer.")
            
            # Find matching module
            matching_module = None
            for name, filepath, module in techniques:
                if name == tech_name:
                    matching_module = module
                    break
                    
            if matching_module is None:
                raise ValueError(f"Could not resolve module for technique: {tech_name}")
                
            print(f"\n[*] Generating {ratio}x expanded dataset via '{tech_name}'...")
            
            if ratio == 1:
                df_final = df.copy()
            else:
                chunks = [df]
                for i in range(ratio - 1):
                    print(f"    -> Synthesizing chunk {i + 1}/{ratio - 1}...")
                    df_augmented = matching_module.augment_dataframe(df, target_col)
                    # The synthetic part is the last len(df) rows
                    synth_chunk = df_augmented.iloc[len(df):].copy()
                    chunks.append(synth_chunk)
                df_final = pd.concat(chunks, ignore_index=True)
                
            export_data = {
                'technique': tech_name,
                'ratio': ratio,
                'dataframe': df_final
            }
            with open("selected_augmentation.pkl", "wb") as f:
                pickle.dump(export_data, f)
                
            print(f"[OK] Successfully synthesized {len(df_final)} rows.")
            print("[OK] Augmented dataset exported to 'selected_augmentation.pkl'!\n")
            
        print("[#] Evaluation successfully completed! Benchmark is idle.\n")
        
    except Exception as e:
        print(f"\n[!] Critical System Error during evaluation execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
