import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class PayloadChangeInjector(BaseInjector):
    """
    Scenario D: Payload/Propagation (FAULT_SYS)
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int = None):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_SYS'
        self.has_printed = False
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> Injecting {self.fault_name}: System-wide payload collision (all joints)...")
            self.has_printed = True
            
        all_current_cols = feature_map.get('actual_current', [])
        
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        severity = np.ones(window_size) * 1.0
        current_multiplier = 1.5
        
        for col in all_current_cols:
            if col and col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] * current_multiplier
                
        tcp_cols = feature_map.get('tcp_pose', [])
        for col in tcp_cols:
            if col and col in df.columns:
                df.loc[mask, col] += np.random.normal(0, 0.05, window_size)
                
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
