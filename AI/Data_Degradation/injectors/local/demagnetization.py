import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class DemagnetizationInjector(BaseInjector):
    """
    Scenario E: Permanent Magnet Demagnetization (FAULT_MD)
    Physics-Informed: Coupled Kt Decay and Thermal Vicious Cycle.
    Signature: Torque-Current Divergence Gap and Position-Dependent Torque Ripple.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int, mode: str = 'incipient'):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_MD'
        self.has_printed = False
        self.mode = mode # 'incipient' or 'abrupt'
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> [ELECTROMAGNETIC] Injecting {self.fault_name} ({self.mode.upper()}): Kt Decay on Joint {self.joint_index}...")
            self.has_printed = True
            
        # Target channels
        cur_actual_col = feature_map.get('actual_current', [])[self.joint_index]
        trq_target_col = feature_map.get('target_moment', [])[self.joint_index]
        temp_col = feature_map.get('joint_temperature', [])[self.joint_index]
        pos_actual_col = feature_map.get('actual_q', [])[self.joint_index]
        vel_actual_col = feature_map.get('actual_qd', [])[self.joint_index]
        
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # 1. Torque Constant (Kt) Degradation Profile
        t_chunk = np.linspace(0, 1, duration)
        raw_severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='exponential')
        
        if self.mode == 'abrupt':
            # Sudden step-drop modeling a short-circuit event
            # Kt drops to 60% of nominal instantly
            kt_profile = 1.0 - (0.4 * (raw_severity > 0.05))
        else:
            # Incipient thermal drift
            kt_profile = 1.0 - (0.45 * raw_severity)
            
        # 2. Torque-Current Divergence Gap
        if cur_actual_col and cur_actual_col in df.columns:
            # I_actual = Torque_target / Kt
            # We model this by scaling the current by (1/Kt)
            # Clip Kt to avoid division by zero
            kt_safe = np.maximum(kt_profile, 0.1)
            multiplier = 1.0 / kt_safe
            df.loc[mask, cur_actual_col] *= multiplier
            
        # 3. Thermal Vicious Cycle Feedback
        if temp_col and temp_col in df.columns:
            # Joule heating (I^2 * R) increases exponentially with 1/Kt
            # We add a thermal surge proportional to the current multiplier squared
            thermal_surge = 12.0 * (multiplier - 1.0)**2
            df.loc[mask, temp_col] += thermal_surge
            
        # 4. Position-Dependent Torque Ripple (Harmonic Jitter)
        if vel_actual_col and vel_actual_col in df.columns:
            vel_vals = df.loc[mask, vel_actual_col].values
            if pos_actual_col and pos_actual_col in df.columns:
                pos_vals = df.loc[mask, pos_actual_col].values
                # Harmonic ripple: 8 pole-pairs nominal
                ripple_mag = 0.12 * raw_severity
                ripple = ripple_mag * np.sin(8 * pos_vals)
                df.loc[mask, vel_actual_col] = vel_vals + ripple
                
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, raw_severity)
