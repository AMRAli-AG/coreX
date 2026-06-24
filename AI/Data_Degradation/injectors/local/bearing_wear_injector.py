import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class BearingWearInjector(BaseInjector):
    """
    Scenario BW: Bearing Race-way Wear & Rolling Contact Fatigue (RCF).
    Research Validation: High-frequency vibrational analysis of rotary actuators.
    
    Symptoms:
    - High-frequency jitter in actual velocity (actual_qd).
    - Torque Ripple: Periodic fluctuations in motor current.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BW'
        self.vibe_amplitude = 0.08  # rad/s noise magnitude
        self.ripple_freq = 0.5      # normalized frequency

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # Polynomial severity for bearing fatigue
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='polynomial')
        
        vel_col = feature_map.get('actual_qd', [])[self.joint_index]
        cur_col = feature_map.get('actual_current', [])[self.joint_index]
        
        # 1. RCF Jitter (Broadband Vibrational Signature)
        if vel_col in df.columns:
            # Add stochastic jitter to velocity
            noise = np.random.normal(0, self.vibe_amplitude, duration) * severity
            df.loc[mask, vel_col] += noise
            
        # 2. Torque Ripple (Degraded Cage Dynamics)
        if cur_col in df.columns:
            time_steps = np.arange(duration)
            # Periodic ripple modulating the current
            ripple = (1.0 + 0.2 * np.sin(self.ripple_freq * time_steps) * severity)
            df.loc[mask, cur_col] *= ripple
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
