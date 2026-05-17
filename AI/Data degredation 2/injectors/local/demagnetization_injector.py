import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class DemagnetizationInjector(BaseInjector):
    """
    Scenario MD: Permanent Magnet Synchronous Motor (PMSM) Demagnetization.
    Research Validation: Flux decay in High-Torque Robotic Actuators.
    
    Mathematical Model:
    - τ = Kt * I
    - Fault: Kt_degraded = Kt_nominal * (1 - decay_factor * Severity)
    - Consequence: To maintain τ, I must increase: I_degraded = I_baseline / (1 - decay)
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int, mode: str = 'incipient'):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_MD'
        self.max_kt_decay = 0.45  # 45% loss of magnetic flux
        self.mode = mode

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # Mode-based severity selection
        profile = 'linear' if self.mode == 'incipient' else 'abrupt'
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile=profile)
        
        cur_col = feature_map.get('actual_current', [])[self.joint_index]
        temp_col = feature_map.get('joint_temperature', [])[self.joint_index]
        
        # 1. Kt Decay -> Inverse Current Elevation
        if cur_col in df.columns:
            # Kt reduction factor
            kt_retention = 1.0 - (self.max_kt_decay * severity)
            # I_req = I_nom / Kt_retention
            df.loc[mask, cur_col] /= kt_retention
            
        # 2. Thermal Efficiency Loss (Higher Joule heating for same torque)
        if temp_col in df.columns:
            # Efficiency drop leads to heat build-up
            thermal_load = (10.0 * severity)
            df.loc[mask, temp_col] += thermal_load
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
