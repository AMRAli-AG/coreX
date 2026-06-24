import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class EncoderDriftInjector(BaseInjector):
    """
    Scenario ED: Encoder Feedback Drift (Optical Scale Contamination).
    Research Validation: Sensor bias and feedback measurement degradation.
    
    Symptoms:
    - Linear or Random-Walk drift in actual position (actual_q).
    - Resulting Positional Error (Actual vs Target) grows over time.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_ED'
        self.drift_bias = 0.05  # rad (max drift offset)

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='linear')
        
        pos_col = feature_map.get('actual_q', [])[self.joint_index]
        
        # 1. Measurement Bias (The "Drift")
        if pos_col in df.columns:
            # Linear drift signal
            time_steps = np.arange(duration)
            # Drift grows with time and severity
            drift_signal = (time_steps / duration) * self.drift_bias * severity
            df.loc[mask, pos_col] += drift_signal
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
