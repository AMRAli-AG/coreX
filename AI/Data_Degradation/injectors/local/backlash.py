import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class BacklashInjector(BaseInjector):
    """
    Scenario A: Gear Backlash (FAULT_BL)
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BL'
        self.has_printed = False
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        # Get overlap indices for this chunk
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> Injecting {self.fault_name}: Backlash positional lag active on Joint {self.joint_index}...")
            self.has_printed = True
            
        # Identify columns
        target_q_col = feature_map['target_q'][self.joint_index]
        actual_q_col = feature_map['actual_q'][self.joint_index]
        target_qd_col = feature_map.get('target_qd', [None] * (self.joint_index+1))[self.joint_index]
        
        if target_q_col is None or actual_q_col is None:
            return df
            
        # Calculate velocity if target_qd is not available
        if target_qd_col is None or target_qd_col not in df.columns:
            target_qd = df[target_q_col].diff().fillna(0)
        else:
            target_qd = df[target_qd_col]
            
        # Create mask for the injection window within THIS chunk
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        # Severity calculation globally across the entire fault window
        total_duration = self.end_idx - self.start_idx
        relative_start = overlap_start - self.start_idx
        time_steps = np.linspace(relative_start/total_duration, (relative_start+window_size)/total_duration, window_size, endpoint=False)
        severity = np.exp(time_steps * 4.6) / 100.0  # Approx 0.01 to 1.0
        
        # Identify direction changes (velocity sign flips)
        sign_flips = np.sign(target_qd).diff().ne(0) & (target_qd != 0)
        
        max_backlash = 0.05 
        
        lag_effect = np.zeros(len(df))
        lag_effect[mask] = sign_flips[mask] * max_backlash * severity
        
        lag_effect = pd.Series(lag_effect).rolling(window=5, min_periods=1).mean()
        
        df.loc[mask, actual_q_col] = df.loc[mask, actual_q_col] - lag_effect[mask] * np.sign(target_qd[mask])
        
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
