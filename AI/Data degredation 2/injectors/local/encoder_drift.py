import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class EncoderDriftInjector(BaseInjector):
    """
    Scenario F: Encoder Drift & Cable Fatigue (FAULT_ED)
    Physics-Informed: Feedback Measurement Blindness and Phantom Control.
    Signature: Cumulative Positional Bias, Erratic Current Corrections, and Categorical Error Latching.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_ED'
        self.has_printed = False
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> [INSTRUMENTATION] Injecting {self.fault_name}: Feedback Blindness & Cable Fatigue on Joint {self.joint_index}...")
            self.has_printed = True
            
        # Target channels
        pos_actual_col = feature_map.get('actual_q', [])[self.joint_index]
        cur_actual_col = feature_map.get('actual_current', [])[self.joint_index]
        
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='exponential')
        
        # 1. Cumulative Feedback Drift (actual_q)
        if pos_actual_col and pos_actual_col in df.columns:
            # Linear/Exponential accelerating bias
            # Max drift: 0.05 rad (significant for a 6-DOF arm)
            drift_bias = 0.05 * severity
            df.loc[mask, pos_actual_col] += drift_bias
            
        # 2. Actuator Phantom Correction Fluctuations (actual_current)
        if cur_actual_col and cur_actual_col in df.columns:
            # Erratic oscillations as PID tries to correct the fake error
            # Magnitude increases with the drift magnitude
            correction_jitter = np.random.normal(0, 0.4 * severity, duration)
            # Add some 'jerks' (low-frequency bursts)
            t_axis = np.linspace(0, duration/125, duration)
            jerks = 0.8 * severity * np.sin(2 * np.pi * 3 * t_axis) * (np.random.rand(duration) > 0.95)
            
            df.loc[mask, cur_actual_col] += (correction_jitter + jerks)
            
        # 3. Discrete State Error Code Injection (robot_mode)
        # We assume 'robot_mode' is the column for categorical status
        mode_col = 'robot_mode' if 'robot_mode' in df.columns else None
        if not mode_col:
            # Fallback to search for any 'mode' or 'status' column
            cols = [c for c in df.columns if 'mode' in c.lower() or 'status' in c.lower()]
            if cols: mode_col = cols[0]
            
        if mode_col:
            # Intermittent bursts in Phase 1, permanent latch in Phase 2 (severity > 0.6)
            err_code = "ERROR_ENC_FAIL"
            # Probability of error flag based on severity
            prob_flag = severity**2 
            error_mask = (np.random.rand(duration) < prob_flag) | (severity > 0.7)
            
            # Since we can't easily mix strings and floats in a single pandas update if it was float
            # we ensure the column is object type first
            df[mode_col] = df[mode_col].astype(object)
            df.loc[mask, mode_col] = np.where(error_mask, err_code, df.loc[mask, mode_col])
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
