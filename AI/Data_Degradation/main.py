import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import time

from augmentation.expander import DataExpander
from utils.data_manager import DataManager
from injectors.local.backlash import BacklashInjector
from injectors.local.friction import LubricationFailureInjector
from injectors.local.bearing_wear import BearingWearInjector
from injectors.propagation.payload_collision import PayloadChangeInjector
from injectors.combos.combo import ComboInjector

def print_banner():
    print("="*70)
    print("   INTEGRATED ROBOTIC FAULT INJECTION & AUGMENTATION PIPELINE   ")
    print("="*70)

def plot_results(output_file, target_col, start_idx, duration):
    print("\nGenerating visualization...")
    df = pd.read_csv(output_file, skiprows=lambda x: x > 0 and (x < start_idx - 1000 or x > start_idx + duration + 1000))
    # Note: skiprows is 1-indexed for the data, but header is row 0.
    # A safer way for smaller datasets is to read all or chunk to find the window.
    # Since it might be large, let's just read the specific window using indexing if possible.
    # Actually pandas read_csv with skiprows can be tricky. Let's just load the whole file if it's not too huge, 
    # or read only the chunk around the fault.
    
    # For robust plotting, let's just read it directly:
    df = pd.read_csv(output_file)
    
    if target_col not in df.columns:
        print(f"Column {target_col} not found for plotting.")
        return
        
    window_df = df.iloc[max(0, start_idx - 500) : min(len(df), start_idx + duration + 500)]
    
    plt.figure(figsize=(12, 6))
    
    healthy = window_df[window_df['fault_label'] == 'Healthy']
    faulty = window_df[window_df['fault_label'] != 'Healthy']
    
    plt.plot(window_df.index, window_df[target_col], label='Signal Trace', color='gray', alpha=0.5)
    plt.scatter(healthy.index, healthy[target_col], color='blue', s=10, label='Healthy')
    if not faulty.empty:
        plt.scatter(faulty.index, faulty[target_col], c=faulty['severity_score'], cmap='YlOrRd', s=15, label='Fault Injection (Severity)')
        plt.colorbar(label='Severity Score')
        
    plt.title(f"Signal Degradation Visualization: {target_col}")
    plt.xlabel("Time Step (Row Index)")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print_banner()
    
    base_file = "ready_for_ai.csv"
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found in the current directory.")
        sys.exit(1)
        
    # Analysis
    print("\n[Stage 0] Analyzing Dataset...")
    df_head = pd.read_csv(base_file, nrows=5)
    total_rows = sum(1 for row in open(base_file, 'r')) - 1
    
    table = [
        ["File Name", base_file],
        ["Total Rows", total_rows],
        ["Total Columns", len(df_head.columns)]
    ]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Stage 1: Augmentation
    print("\n" + "="*40)
    print(" STAGE 1: Data Augmentation & Expansion ")
    print("="*40)
    
    expand_choice = input("Do you want to expand the dataset size? (Yes/No): ").strip().lower()
    filepath = base_file
    
    if expand_choice in ['yes', 'y']:
        factor = int(input("Enter expansion factor (e.g., 2, 5, 10): "))
        expander = DataExpander(base_file)
        expanded_df = expander.expand(factor)
        filepath = f"expanded_{factor}x_{base_file}"
        expanded_df.to_csv(filepath, index=False)
        print(f"Expanded dataset saved to: {filepath}")
        
    # Stage 2: Fault Injection
    print("\n" + "="*40)
    print(" STAGE 2: Realistic Fault Injection ")
    print("="*40)
    
    print("Available Diseases:")
    print(" BL    : Backlash (Positional Lag)")
    print(" LF    : Lubrication Failure (Friction/Thermal)")
    print(" BW    : Bearing Wear (High-frequency Noise & Peaks)")
    print(" SYS   : System Propagation (Payload Collision)")
    print(" COMBO : Multiple Faults (e.g., BL + BW)")
    
    fault_choice = input("Which fault would you like to inject? (BL/LF/BW/SYS/COMBO): ").strip().upper()
    
    joint_idx = 0
    if fault_choice in ['BL', 'LF', 'BW', 'COMBO']:
        joint_idx = int(input("Enter Target Joint Index (e.g., 0 for Joint 1): "))
        
    start_idx = int(input("Enter starting timestamp/row for degradation (e.g., 5000): "))
    duration = int(input("Enter duration of the fault (e.g., 10000): "))
    end_idx = start_idx + duration
    
    print("\nInitializing Injector...")
    if fault_choice == 'BL':
        injector = BacklashInjector(start_idx, end_idx, joint_idx)
        target_col_key = 'actual_q'
    elif fault_choice == 'LF':
        injector = LubricationFailureInjector(start_idx, end_idx, joint_idx)
        target_col_key = 'actual_current'
    elif fault_choice == 'BW':
        injector = BearingWearInjector(start_idx, end_idx, joint_idx)
        target_col_key = 'actual_current'
    elif fault_choice == 'SYS':
        injector = PayloadChangeInjector(start_idx, end_idx)
        target_col_key = 'actual_current'
        joint_idx = 0 # Default for plotting
    elif fault_choice == 'COMBO':
        inj1 = BacklashInjector(start_idx, end_idx, joint_idx)
        inj2 = BearingWearInjector(start_idx, end_idx, joint_idx)
        injector = ComboInjector(start_idx, end_idx, joint_idx, [inj1, inj2])
        target_col_key = 'actual_current'
    else:
        print("Invalid fault choice.")
        sys.exit(1)
        
    output_file = "labeled_robot_data.csv"
    dm = DataManager(filepath)
    dm.process_and_save(injector, output_file)
    
    # Summary Report
    print("\n" + "="*40)
    print(" FINAL SUMMARY REPORT ")
    print("="*40)
    
    final_rows = sum(1 for row in open(output_file, 'r')) - 1
    report = [
        ["Total Rows Generated", final_rows],
        ["Fault Type", injector.fault_name],
        ["Target Joint", joint_idx if fault_choice != 'SYS' else "ALL"],
        ["Injection Window", f"Row {start_idx} to {end_idx}"],
        ["Output File", output_file]
    ]
    print(tabulate(report, headers=["Metric", "Details"], tablefmt="grid"))
    
    # Find exact column name for plotting
    col_name = dm.feature_map.get(target_col_key, [])[joint_idx]
    if col_name:
        plot_results(output_file, col_name, start_idx, duration)
    else:
        print("Could not find appropriate column to visualize.")

if __name__ == "__main__":
    main()
