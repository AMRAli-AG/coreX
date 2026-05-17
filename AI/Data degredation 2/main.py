import os
import sys

import pandas as pd

from augmentation.expander import DataExpander
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
    print("\n--- LOCALIZED FAULT PROFILE CATALOG (14) ---")
    for i, profile in enumerate(LOCAL_FAULT_PROFILES, 1):
        joints = "ALL" if profile.distributed else str(list(profile.eligible_joints))
        mode = f" | mode={profile.mode}" if profile.mode else ""
        print(
            f"  {i:2d}. {profile.profile_id:<16} {profile.fault_key} "
            f"[{profile.implementation}]{mode} -> joints {joints}"
        )
    print("\n--- PROPAGATION PROFILES ---")
    for profile in PROPAGATION_FAULT_PROFILES:
        print(f"      {profile.profile_id:<16} {profile.fault_key}")
    print()


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

    print("--- STAGE 1: PHYSICS-INFORMED DATA EXPANSION ---".center(80))
    expand_choice = input("[?] Apply dataset expansion? (y/n): ").strip().lower()
    filepath = base_file

    if expand_choice in ['yes', 'y']:
        factor = int(input("[>] Expansion factor: "))
        method = input("[>] Method (gan/warp/basic): ").strip().lower()

        expander = DataExpander(base_file)
        expanded_df = expander.expand(factor, method=method)

        filepath = f"expanded_{factor}x_{method}_{base_file}"
        expanded_df.to_csv(filepath, index=False)
        viz.plot_stitching_dashboard(expanded_df, total_rows, dm.feature_map, joint_idx=1)

    print("\n" + "--- STAGE 2: REGISTRY-BACKED FAULT INJECTION ---".center(80))
    print_profile_catalog()
    print(
        "Modes: [profile_id] e.g. BL_FULL | Legacy [BL|LF|BW|EE|MD|ED|SYS] | "
        "[COMBO] | [CASCADE] | [ALL] full 14-profile suite"
    )

    fault_choice = input("[?] Select profile_id, disease code, COMBO, CASCADE, or ALL: ").strip().upper()

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


def _save_and_visualize(filepath, injector, df_temp, start_idx, duration, fault_key, joint_idx, is_distributed, viz):
    dm_proc = DataManager(filepath)
    output_file = "labeled_robot_data.csv"
    dm_proc.process_and_save(injector, output_file)

    final_df = pd.read_csv(output_file)
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
