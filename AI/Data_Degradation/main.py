import os
import sys

import pandas as pd

from injectors.base_injector import BaseInjector
from injectors.fault_registry import (
    LOCAL_FAULT_PROFILES,
    build_injector,
    get_profile,
    iter_local_profiles,
)
from injectors.combos.combo import ComboInjector
from injectors.fault_registry import ALL_FAULT_PROFILES, PROPAGATION_FAULT_PROFILES
from utils.data_manager import DataManager
from utils.fault_kb import FaultKB
from utils.phm_engine import PHMEngine
from utils.visualizer import Visualizer


def _profile_eligibility_table() -> dict:
    """Build interactive eligibility map from the 14-profile registry."""
    table = {}
    for profile in ALL_FAULT_PROFILES:
        code = profile.fault_key.replace('FAULT_', '')
        if code not in table:
            table[code] = {
                'name': profile.name.split('—')[0].split('(')[0].strip(),
                'eligible': list(profile.eligible_joints),
                'reasoning': profile.reasoning,
                'profiles': [],
                'distributed': profile.distributed,
            }
        table[code]['profiles'].append(profile.profile_id)
        if profile.distributed:
            table[code]['distributed'] = True
    return table


FAULT_ELIGIBILITY = _profile_eligibility_table()


def print_banner():
    print("\n" + "=" * 80)
    print("   RESEARCH-VALIDATED PHM FRAMEWORK v3.1 | 6-DOF ROBOTIC ANALYTICS   ")
    print("=" * 80)
    print("   IEEE/PHM Standards | 14 Localized Profiles | Registry-Driven Injection")
    print("=" * 80 + "\n")


def print_profile_catalog():
    print("\n" + "="*100)
    print("                      LOCALIZED FAULT PROFILE CATALOG (14 Standard Modes)                      ")
    print("="*100)
    print(f" {'Idx':<4} | {'Profile ID':<15} | {'Fault Description':<42} | {'Eligible Joints':<15} | {'Tier':<8}")
    print("-"*100)
    for i, profile in enumerate(LOCAL_FAULT_PROFILES, 1):
        joints = "All Joints" if profile.distributed else f"Joints {list(profile.eligible_joints)}"
        tier = profile.implementation.upper()
        # Clean name for table display
        clean_name = profile.name.split('—')[0].split('(')[0].strip()
        if profile.mode:
            clean_name += f" ({profile.mode.capitalize()})"
        print(f"  {i:<2d}  | {profile.profile_id:<15} | {clean_name:<42} | {joints:<15} | {tier:<8}")
    print("="*100)
    print("                      PROPAGATION & SYSTEM-WIDE FAULT MODES                      ")
    print("="*100)
    for profile in PROPAGATION_FAULT_PROFILES:
        print(f"  SYS  | {profile.profile_id:<15} | {profile.name:<42} | All Joints (Prop) | FULL")
    print("="*100 + "\n")


def check_idempotency(filepath: str) -> bool:
    df_preview = pd.read_csv(filepath, nrows=200)
    if 'fault_label' in df_preview.columns:
        unique_labels = df_preview['fault_label'].unique()
        if any(label != 'Healthy' for label in unique_labels):
            print(f"\n[WARNING] Dataset '{filepath}' already contains active fault labels.")
            return False
    return True


def resolve_injector(
    fault_choice: str,
    start_idx: int,
    end_idx: int,
    joint_idx: int | None,
    profile_id: str | None = None,
    mode: str | None = None,
) -> BaseInjector:
    """Resolve a single injector from registry profile_id or legacy disease code."""
    if profile_id:
        profile = get_profile(profile_id)
        if profile.fault_key == 'FAULT_MD' and mode and profile.profile_id in (
            'MD_INC_FULL', 'MD_ABR_FULL', 'MD_COMPACT'
        ):
            return profile.build(start_idx, end_idx, joint_idx, mode_override=mode)
        if profile.fault_key == 'FAULT_MD' and mode == 'abrupt' and profile.profile_id == 'MD_INC_FULL':
            return get_profile('MD_ABR_FULL').build(start_idx, end_idx, joint_idx)
        return profile.build(start_idx, end_idx, joint_idx)

    fault_key = f"FAULT_{fault_choice}"

    if fault_choice == 'COMBO':
        combo_input = input("[>] Combo profile IDs (comma-separated, e.g. BL_FULL,BW_FULL): ").strip()
        ids = [p.strip() for p in combo_input.split(',') if p.strip()]
        return ComboInjector.from_profile_ids(ids, start_idx, end_idx, joint_idx)

    if fault_choice == 'CASCADE':
        primary = input("[>] Primary profile_id (e.g. LF_DIST_FULL): ").strip()
        return ComboInjector.cascading_from_primary(primary, start_idx, end_idx, joint_idx)

    if fault_choice == 'SYS':
        return build_injector('SYS_COLLISION', start_idx, end_idx)

    if fault_choice == 'MD' and mode:
        pid = 'MD_ABR_FULL' if mode == 'abrupt' else 'MD_INC_FULL'
        return build_injector(pid, start_idx, end_idx, joint_idx)

    default_full = {
        'BL': 'BL_FULL',
        'LF': 'LF_DIST_FULL',
        'BW': 'BW_FULL',
        'EE': 'EE_FULL',
        'MD': 'MD_INC_FULL',
        'ED': 'ED_FULL',
    }
    if fault_choice in default_full:
        return build_injector(default_full[fault_choice], start_idx, end_idx, joint_idx)

    raise ValueError(f"No registry mapping for fault choice '{fault_choice}' ({fault_key})")


def run_full_suite(filepath: str, output_dir: str = "labeled_outputs") -> None:
    """
    Batch-generate labeled datasets for all 14 localized degradation profiles.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_temp = pd.read_csv(filepath)
    total_len = len(df_temp)

    print(f"\n[*] Running full 14-profile localized injection suite on {filepath}")
    print(f"    Output directory: {output_dir}\n")

    for profile in iter_local_profiles():
        fault_key = profile.fault_key
        start_idx, duration = PHMEngine.calculate_window(total_len, fault_key)
        end_idx = start_idx + duration

        joint = None if profile.distributed else profile.eligible_joints[0]
        injector = profile.build(start_idx, end_idx, joint)

        out_name = f"{profile.profile_id}_labeled.csv"
        out_path = os.path.join(output_dir, out_name)

        dm = DataManager(filepath)
        dm.process_and_save(injector, out_path)
        print(f"    [OK] {profile.profile_id} -> {out_path}\n")

    print(f"[#] Suite complete: {len(LOCAL_FAULT_PROFILES)} datasets written to {output_dir}/")


def main():
    print_banner()
    viz = Visualizer()

    base_file = "ready_for_ai.csv"
    if not os.path.exists(base_file):
        print(f"[!] Critical Error: {base_file} not found.")
        sys.exit(1)

    if not check_idempotency(base_file):
        sys.exit(1)

    dm = DataManager(base_file)
    total_rows = sum(1 for _ in open(base_file, 'r')) - 1

    print("--- STAGE 1: DYNAMIC DATA AUGMENTATION EVALUATION ---".center(80))
    eval_choice = input("[?] Run the Dynamic Augmentation Evaluation and Benchmarking Suite? (y/n): ").strip().lower()
    filepath = base_file

    if eval_choice in ['yes', 'y']:
        import evaluate_augmentations
        evaluate_augmentations.main()
        print(f"\n[*] Dynamic evaluation complete. Benchmark results stored in 'augmentation_accuracy.csv'.\n")

    # Integrate Stage 2 loading from pickled Stage 1 selections
    pkl_file = "selected_augmentation.pkl"
    if os.path.exists(pkl_file):
        print("\n" + "--- STAGE 1 EXPORT DETECTED ---".center(80))
        load_aug_choice = input("[?] Detected 'selected_augmentation.pkl'. Use this augmented dataset for Stage 2? (y/n) [y]: ").strip().lower() or 'y'
        
        if load_aug_choice in ['yes', 'y']:
            import pickle
            try:
                with open(pkl_file, 'rb') as f:
                    export_data = pickle.load(f)
                tech = export_data.get('technique', 'Unknown')
                ratio = export_data.get('ratio', 1)
                df_aug = export_data['dataframe']
                
                # Export to CSV for Stage 2 consumption
                filepath = "selected_augmented_data.csv"
                df_aug.to_csv(filepath, index=False)
                
                print(f"[OK] Successfully loaded '{tech}' with {ratio}x ratio.")
                print(f"[OK] Temporary dataset written to '{filepath}' ({len(df_aug)} rows).")
            except Exception as e:
                print(f"[!] Error loading pickled dataset: {e}. Falling back to '{base_file}'.")
                filepath = base_file
        else:
            print(f"[*] Proceeding with the original baseline dataset: '{base_file}'.")
            filepath = base_file

    print("\n" + "--- STAGE 2: REGISTRY-BACKED FAULT INJECTION ---".center(80))
    print_profile_catalog()
    
    print("Testing Capabilities Guide:")
    print("  -> Single Fault: Enter a Profile ID (e.g., BL_FULL) OR its Index Number (1-14) from the table.")
    print("  -> [COMBO]      : Simulate multiple overlapping faults simultaneously on a representative joint.")
    print("  -> [CASCADE]    : Simulate a sequential failure scenario triggered by a primary joint fault.")
    print("  -> [ALL]        : Batch-execute the entire 14-profile localized failure catalog automatically.")
    print("  -> [SYS]        : Inject a system-wide payload/collision shock across all motor currents.")
    print("-" * 100)

    fault_choice = input("[?] Select fault (ID, Index 1-14, COMBO, CASCADE, ALL, or SYS): ").strip()

    if fault_choice.isdigit():
        idx = int(fault_choice)
        if 1 <= idx <= len(LOCAL_FAULT_PROFILES):
            profile_id = LOCAL_FAULT_PROFILES[idx - 1].profile_id
            fault_choice = profile_id
            print(f"[+] Index resolved to Profile ID: '{profile_id}'")
        else:
            print(f"[!] Invalid index choice '{idx}'. Please enter a number between 1 and {len(LOCAL_FAULT_PROFILES)}.")
            sys.exit(1)
    else:
        fault_choice = fault_choice.upper()

    if fault_choice == 'ALL':
        run_full_suite(filepath)
        return

    df_temp = pd.read_csv(filepath)

    if fault_choice == 'COMBO':
        combo_input = input("[>] Combo profile IDs (comma-separated): ").strip()
        ids = [p.strip() for p in combo_input.split(',') if p.strip()]
        joint_idx = int(input("[>] Representative joint index: ") or "1")
        start_idx, duration = PHMEngine.calculate_window(len(df_temp), 'FAULT_BL')
        injector = ComboInjector.from_profile_ids(ids, start_idx, start_idx + duration, joint_idx)
        _save_and_visualize(filepath, injector, df_temp, start_idx, duration, 'FAULT_COMBO', joint_idx, False, viz)
        return

    if fault_choice == 'CASCADE':
        primary = input("[>] Primary profile_id (e.g. LF_DIST_FULL): ").strip()
        profile = get_profile(primary)
        joint_idx = None if profile.distributed else int(input(f"[>] Joint {list(profile.eligible_joints)}: "))
        start_idx, duration = PHMEngine.calculate_window(len(df_temp), profile.fault_key)
        injector = ComboInjector.cascading_from_primary(primary, start_idx, start_idx + duration, joint_idx)
        _save_and_visualize(
            filepath, injector, df_temp, start_idx, duration, profile.fault_key, joint_idx or 0,
            profile.distributed, viz
        )
        return

    profile_id = None
    if fault_choice in {p.profile_id for p in ALL_FAULT_PROFILES}:
        profile_id = fault_choice
        profile = get_profile(profile_id)
        fault_key = profile.fault_key
        meta = {
            'name': profile.name,
            'eligible': list(profile.eligible_joints),
            'reasoning': profile.reasoning,
            'distributed': profile.distributed,
        }
        legacy_code = fault_key.replace('FAULT_', '')
    elif fault_choice in FAULT_ELIGIBILITY:
        legacy_code = fault_choice
        meta = FAULT_ELIGIBILITY[fault_choice]
        fault_key = f"FAULT_{fault_choice}"
    else:
        print("[!] Invalid selection.")
        sys.exit(1)

    print(f"\n[RESEARCH CHECK] {meta['name']}:")
    print(f"   Eligible Joints: {meta['eligible']}")
    print(f"   Rationale: {meta['reasoning']}")
    if 'profiles' in meta:
        print(f"   Registry profiles: {meta['profiles']}")

    is_distributed = meta.get('distributed', False)
    joint_idx = None
    mode = None

    if fault_choice not in ('COMBO', 'CASCADE') and not is_distributed:
        while True:
            try:
                joint_idx = int(input(f"[>] Select Target Joint ({meta['eligible']}): "))
                if joint_idx in meta['eligible']:
                    break
                print(f"\n[BLOCK] Joint {joint_idx} is physically ineligible for {meta['name']}.")
                print(f"[WHY?] {meta['reasoning']}\n")
            except ValueError:
                print("[!] Please enter a valid joint index.")
    elif is_distributed:
        print("[*] Distributed fault: targeting all joints per registry profile.")

    if legacy_code == 'MD' or (profile_id and profile_id.startswith('MD')):
        mode = input("[>] Select Mode (incipient/abrupt) [incipient]: ").strip().lower() or 'incipient'

    start_idx, duration = PHMEngine.calculate_window(len(df_temp), fault_key)
    end_idx = start_idx + duration

    injector = resolve_injector(
        legacy_code,
        start_idx,
        end_idx,
        joint_idx,
        profile_id=profile_id,
        mode=mode,
    )

    _save_and_visualize(filepath, injector, df_temp, start_idx, duration, fault_key, joint_idx, is_distributed, viz)


def verify_injection_signature(
    healthy_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    start_idx: int,
    duration: int,
    fault_key: str,
    joint_idx: int | None,
    is_distributed: bool
) -> None:
    """
    Perform a physical/statistical mechatronic sanity check to audit the integrity
    of the injected failure signature.
    """
    end_idx = start_idx + duration
    
    # Extract fault window regions
    h_window = healthy_df.iloc[start_idx:end_idx]
    d_window = degraded_df.iloc[start_idx:end_idx]
    
    # 1. Structural checks
    has_labels = 'fault_label' in degraded_df.columns
    has_severity = 'severity_score' in degraded_df.columns
    
    max_severity = d_window['severity_score'].max() if has_severity else 0.0
    active_labels = d_window['fault_label'].unique() if has_labels else []
    
    print("\n" + "=" * 100)
    print("                      PHYSICAL & STATISTICAL SIGNATURE INTEGRITY AUDIT                       ")
    print("=" * 100)
    
    print(f"  [METADATA] Target Fault Profile : {fault_key}")
    print(f"  [METADATA] Active Window         : Rows {start_idx} to {end_idx} ({duration} samples)")
    print(f"  [METADATA] Labeling Column check  : {'OK (fault_label present)' if has_labels else 'ERROR'}")
    print(f"  [METADATA] Severity Column check: {'OK (severity_score present)' if has_severity else 'ERROR'}")
    print(f"  [METADATA] Observed Labels       : {list(active_labels)}")
    print(f"  [METADATA] Peak Severity Score   : {max_severity:.4f}")
    print("-" * 100)
    
    # 2. Residual & physical metrics checks
    print(f" {'Telemetry Residual Metric':<40} | {'Baseline (Healthy)':<20} | {'Observed (Faulty)':<20} | {'Status':<8}")
    print("-" * 100)
    
    # Resolve target joints
    joints = range(6) if is_distributed or joint_idx is None else [joint_idx]
    
    # Helper to calculate average values safely
    def get_avg(df, col_prefix, j):
        cols = [c for c in df.columns if col_prefix in c]
        if j < len(cols) and cols[j] in df.columns:
            return df[cols[j]].mean()
        return 0.0
        
    def get_std(df, col_prefix, j):
        cols = [c for c in df.columns if col_prefix in c]
        if j < len(cols) and cols[j] in df.columns:
            return df[cols[j]].std()
        return 0.0

    success = True
    
    for j in joints:
        joint_tag = f"Joint {j}" if not is_distributed else f"Joint {j} (Dist)"
        
        if fault_key == 'FAULT_LF' or 'LF' in fault_key:
            h_curr = get_avg(h_window, 'actual_current', j)
            d_curr = get_avg(d_window, 'actual_current', j)
            h_temp = get_avg(h_window, 'joint_temperature', j)
            d_temp = get_avg(d_window, 'joint_temperature', j)
            
            curr_pct = ((d_curr / h_curr) - 1.0) * 100.0 if h_curr != 0 else 0.0
            temp_rise = d_temp - h_temp
            
            status = "OK" if (curr_pct > 1.0 or temp_rise > 0.5) else "LOW"
            if status == "LOW": success = False
            
            print(f"  [{joint_tag}] Average Phase Current (RMS)   | {h_curr:<18.4f} A | {d_curr:<18.4f} A | {status}")
            print(f"  [{joint_tag}] Joint Core Temperature        | {h_temp:<18.4f} C | {d_temp:<18.4f} C | {status}")
            
        elif fault_key == 'FAULT_BL' or 'BL' in fault_key:
            h_pos = get_avg(h_window, 'actual_q', j)
            d_pos = get_avg(d_window, 'actual_q', j)
            pos_residual = abs(d_pos - h_pos)
            
            status = "OK" if pos_residual > 0.00001 else "LOW"
            if status == "LOW": success = False
            
            print(f"  [{joint_tag}] Mean Positional Feedback (q)   | {h_pos:<18.6f} rad | {d_pos:<18.6f} rad | {status}")
            print(f"  [{joint_tag}] Integrated Backlash Hysteresis | {'0.000000':<18} rad | {pos_residual:<18.6f} rad | {status}")
            
        elif fault_key == 'FAULT_BW' or 'BW' in fault_key:
            h_vstd = get_std(h_window, 'actual_qd', j)
            d_vstd = get_std(d_window, 'actual_qd', j)
            
            status = "OK" if d_vstd > h_vstd else "LOW"
            if status == "LOW": success = False
            
            print(f"  [{joint_tag}] Joint Velocity Jitter (StdDev)| {h_vstd:<18.6f} r/s | {d_vstd:<18.6f} r/s | {status}")
            
        elif fault_key == 'FAULT_EE' or 'EE' in fault_key:
            h_cstd = get_std(h_window, 'actual_current', j)
            d_cstd = get_std(d_window, 'actual_current', j)
            h_temp = get_avg(h_window, 'joint_temperature', j)
            d_temp = get_avg(d_window, 'joint_temperature', j)
            
            temp_rise = d_temp - h_temp
            status = "OK" if (d_cstd > h_cstd or temp_rise > 0.1) else "LOW"
            if status == "LOW": success = False
            
            print(f"  [{joint_tag}] Phase Current Jitter (Arcing) | {h_cstd:<18.4f} A | {d_cstd:<18.4f} A | {status}")
            print(f"  [{joint_tag}] Thermal Step Heat Rise        | {h_temp:<18.4f} C | {d_temp:<18.4f} C | {status}")
            
        elif fault_key == 'FAULT_MD' or 'MD' in fault_key:
            h_curr = get_avg(h_window, 'actual_current', j)
            d_curr = get_avg(d_window, 'actual_current', j)
            
            curr_pct = ((d_curr / h_curr) - 1.0) * 100.0 if h_curr != 0 else 0.0
            status = "OK" if curr_pct > 1.0 else "LOW"
            if status == "LOW": success = False
            
            print(f"  [{joint_tag}] Current Compensation (Kt Decay) | {h_curr:<18.4f} A | {d_curr:<18.4f} A | {status}")
            
        elif fault_key == 'FAULT_SYS' or 'SYS' in fault_key:
            h_curr = get_avg(h_window, 'actual_current', j)
            d_curr = get_avg(d_window, 'actual_current', j)
            
            curr_pct = ((d_curr / h_curr) - 1.0) * 100.0 if h_curr != 0 else 0.0
            status = "OK" if curr_pct > 1.0 else "LOW"
            if status == "LOW": success = False
            
            print(f"  [{joint_tag}] System-wide Phase Shock current | {h_curr:<18.4f} A | {d_curr:<18.4f} A | {status}")

    print("=" * 100)
    if success and has_labels and has_severity:
        print("  [AUDIT RESULT] SIGNATURE INTEGRITY CHECKS PASSED: MECHATRONIC DEGRADATION VALIDATED  ")
    else:
        print("  [AUDIT RESULT] SIGNATURE WARNING: OBSERVED ANOMALY DRIFT IS BELOW STATISTICAL LIMITS ")
    print("=" * 100 + "\n")


def _save_and_visualize(filepath, injector, df_temp, start_idx, duration, fault_key, joint_idx, is_distributed, viz):
    dm_proc = DataManager(filepath)
    output_file = "labeled_robot_data.csv"
    dm_proc.process_and_save(injector, output_file)

    final_df = pd.read_csv(output_file)
    
    # Perform research-grade physical integrity audit
    verify_injection_signature(df_temp, final_df, start_idx, duration, fault_key, joint_idx, is_distributed)

    if is_distributed:
        viz.plot_global_health_map(final_df, df_temp, dm_proc.feature_map, start_idx, duration, fault_key)
        viz.plot_local_diagnostic_dashboard(
            final_df, df_temp, dm_proc.feature_map, 1, start_idx, duration, fault_key
        )
    else:
        viz.plot_local_diagnostic_dashboard(
            final_df, df_temp, dm_proc.feature_map, joint_idx, start_idx, duration, fault_key
        )

    print("\n" + "=" * 80)
    print("   MISSION ACCOMPLISHED: Research-Validated Training Data Generated   ")
    print(f"   Registry: {FaultKB.profile_count()} localized profiles verified")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
