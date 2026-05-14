import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class BearingWearInjector(BaseInjector):
    """
    Scenario C: Bearing Wear (FAULT_BW)
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BW'
        self.has_printed = False
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> Injecting {self.fault_name}: High-frequency stochastic noise (Kurtosis) on Joint {self.joint_index}...")
            self.has_printed = True
            
        actual_qd_col = feature_map['actual_qd'][self.joint_index]
        actual_current_col = feature_map['actual_current'][self.joint_index]
        
        if actual_qd_col is None or actual_current_col is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        total_duration = self.end_idx - self.start_idx
        relative_start = overlap_start - self.start_idx
        severity = np.linspace(max(0.01, relative_start/total_duration), (relative_start+window_size)/total_duration, window_size, endpoint=False)
        
        base_qd = df.loc[mask, actual_qd_col].abs().mean()
        base_curr = df.loc[mask, actual_current_col].abs().mean()
        
        qd_noise = np.random.normal(0, base_qd * 0.1, window_size) * severity
        curr_noise = np.random.normal(0, base_curr * 0.1, window_size) * severity
        
        peak_prob = 0.05 * severity
        peak_mask = np.random.random(window_size) < peak_prob
        
        qd_peaks = peak_mask * np.random.normal(0, base_qd * 0.5, window_size)
        curr_peaks = peak_mask * np.random.normal(0, base_curr * 0.5, window_size)
        
        df.loc[mask, actual_qd_col] += (qd_noise + qd_peaks)
        df.loc[mask, actual_current_col] += (curr_noise + curr_peaks)
        
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
