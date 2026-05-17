import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class ElectricalErosionInjector(BaseInjector):
    """
    Scenario EE: Electrical Erosion (EDM Fluting) of Bearing Surfaces.
    Research Validation: Parasitic bearing currents in VFD-driven robotics.
    
    Symptoms:
    - Step-wise temperature jumps (localized heating).
    - High-frequency arcing spikes in motor current.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_EE'
        self.arc_probability = 0.05  # 5% chance of arcing per sample
        self.arc_magnitude = 2.5     # Current spike multiplier
        self.temp_step = 8.0         # Temperature jump in C

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='abrupt')
        
        cur_col = feature_map.get('actual_current', [])[self.joint_index]
        temp_col = feature_map.get('joint_temperature', [])[self.joint_index]
        
        # 1. High-Frequency Arcing Spikes (Intermittent Current Bursts)
        if cur_col in df.columns:
            # Generate random arcing events
            arc_mask = np.random.random(duration) < (self.arc_probability * severity)
            multipliers = np.ones(duration)
            multipliers[arc_mask] = self.arc_magnitude
            df.loc[mask, cur_col] *= multipliers
            
        # 2. Localized Heating (Thermal Step)
        if temp_col in df.columns:
            # Localized hot-spots from EDM discharges
            df.loc[mask, temp_col] += (self.temp_step * severity)
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
