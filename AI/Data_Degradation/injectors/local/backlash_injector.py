import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class BacklashInjector(BaseInjector):
    """
    Scenario BL: Mechanical Gear Backlash & Surface Wear.
    Research Validation: Non-linear Dead-Zone (2b) switching logic.
    
    Mathematical Model:
    - q_out = q_in - b * sign(v) if |q_in - q_out| > b
    - Fault: b (dead-zone width) increases with severity.
    - Impact: Transient current spikes upon re-engagement.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BL'
        self.nominal_gap = 0.012  # rad (Max dead-zone width)
        self.strike_gain = 1.6    # Multiplier for current re-engagement strike

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # Exponential severity for mechanical wear
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='exponential')
        
        pos_col = feature_map.get('actual_q', [])[self.joint_index]
        vel_col = feature_map.get('actual_qd', [])[self.joint_index]
        cur_col = feature_map.get('actual_current', [])[self.joint_index]
        
        if vel_col in df.columns:
            vel_vals = df.loc[mask, vel_col].values
            signs = np.sign(vel_vals)
            # Detect zero-crossings (reversal boundaries)
            zero_crossings = np.concatenate([[False], np.diff(signs) != 0])
            
            # 1. Kinematic Phase Lag (The "Play" in the gear)
            if pos_col in df.columns:
                pos_vals = df.loc[mask, pos_col].values
                # Modulate position by dead-zone offset
                lag_signal = signs * self.nominal_gap * severity
                df.loc[mask, pos_col] = pos_vals - lag_signal
                
            # 2. Re-engagement Current Strikes
            if cur_col in df.columns:
                cur_vals = df.loc[mask, cur_col].values
                strike_indices = np.where(zero_crossings)[0]
                for idx in strike_indices:
                    # Strike window: 3-5 samples after reversal
                    window_end = min(idx + 5, len(cur_vals))
                    multiplier = 1.0 + (self.strike_gain * (severity[idx] if isinstance(severity, np.ndarray) else severity))
                    cur_vals[idx:window_end] *= multiplier
                df.loc[mask, cur_col] = cur_vals
                
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
