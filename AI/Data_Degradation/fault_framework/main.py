import sys
import os
from data_manager import DataManager
from injectors.local.backlash import BacklashInjector
from injectors.local.friction import LubricationFailureInjector
from injectors.local.bearing_wear import BearingWearInjector
from injectors.propagation.payload_change import PayloadChangeInjector
from injectors.combinations.combo_injector import ComboInjector

def print_banner():
    print("="*60)
    print(" Professional Fault Injection & Data Labeling Framework")
    print("="*60)

def get_user_input():
    filepath = input("Enter dataset path (e.g., ../ready_for_ai.csv): ").strip()
    if not os.path.exists(filepath):
        print("File not found!")
        sys.exit(1)
        
    start_idx = int(input("Enter Starting Time (Row Index, e.g., 5000): "))
    duration = int(input("Enter Duration (Number of Rows, e.g., 15000): "))
    end_idx = start_idx + duration
    
    print("\nAvailable Faults:")
    print("1. Gear Backlash (Local)")
    print("2. Lubrication Failure (Local)")
    print("3. Bearing Wear (Local)")
    print("4. Payload Change (Propagation/System)")
    print("5. Combo (Backlash + Bearing Wear)")
    
    fault_choice = int(input("Select Fault (1-5): "))
    
    joint_idx = 0
    if fault_choice in [1, 2, 3, 5]:
        joint_idx = int(input("Enter Joint Index to inject (e.g., 0, 1, 2...): "))
        
    return filepath, start_idx, end_idx, fault_choice, joint_idx

def main():
    print_banner()
    filepath, start_idx, end_idx, fault_choice, joint_idx = get_user_input()
    
    print("\nAnalyzing features...")
    dm = DataManager(filepath)
    print("Feature Mapping successful based on column names.")
    
    if fault_choice == 1:
        injector = BacklashInjector(start_idx, end_idx, joint_idx)
    elif fault_choice == 2:
        injector = LubricationFailureInjector(start_idx, end_idx, joint_idx)
    elif fault_choice == 3:
        injector = BearingWearInjector(start_idx, end_idx, joint_idx)
    elif fault_choice == 4:
        injector = PayloadChangeInjector(start_idx, end_idx)
    elif fault_choice == 5:
        inj1 = BacklashInjector(start_idx, end_idx, joint_idx)
        inj2 = BearingWearInjector(start_idx, end_idx, joint_idx)
        injector = ComboInjector(start_idx, end_idx, joint_idx, [inj1, inj2])
    else:
        print("Invalid choice")
        sys.exit(1)
        
    output_path = f"injected_{os.path.basename(filepath)}"
    print(f"\nStarting Injection: {injector.fault_name}")
    print(f"Target Joint: {joint_idx}")
    print(f"Time Range: {start_idx} to {end_idx}")
    
    dm.process_and_save(injector, output_path)
    
if __name__ == "__main__":
    main()
