import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class LubricationFailureInjector(BaseInjector):
    """
    Scenario B: Lubrication Failure (FAULT_LF)
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_LF'
        self.has_printed = False
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> Injecting {self.fault_name}: Current-Thermal coupling active on Joint {self.joint_index}...")
            self.has_printed = True
            
        actual_current_col = feature_map['actual_current'][self.joint_index]
        
        temp_cols = feature_map.get('joint_temperature', [])
        temp_col = temp_cols[self.joint_index] if len(temp_cols) > self.joint_index else None
        
        if actual_current_col is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        total_duration = self.end_idx - self.start_idx
        relative_start = overlap_start - self.start_idx
        severity = np.linspace(relative_start/total_duration, (relative_start+window_size)/total_duration, window_size, endpoint=False)
        
        current_multiplier = 1.0 + (0.30 * severity)
        df.loc[mask, actual_current_col] = df.loc[mask, actual_current_col] * current_multiplier
        
        if temp_col and temp_col in df.columns:
            temp_increase = 15.0 * severity
            df.loc[mask, temp_col] = df.loc[mask, temp_col] + temp_increase
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
