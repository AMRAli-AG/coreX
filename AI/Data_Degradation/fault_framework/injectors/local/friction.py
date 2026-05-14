import pandas as pd
import numpy as np
from ..base_injector import BaseInjector

class LubricationFailureInjector(BaseInjector):
    \"\"\"
    Scenario B: Lubrication Failure (FAULT_LF)
    
    Clinical Description:
    Loss of lubrication increases static and dynamic friction within the joint's bearings or gearbox.
    Physically, this forces the motor to draw more current to achieve the same target moment/torque.
    The excess energy lost to friction dissipates as heat, causing an anomalous rise in joint temperature.
    
    Mathematical Logic:
    1. Apply a positive drift to `actual_current` proportional to `target_moment` (or `actual_qd` if moment unavailable).
    2. Apply a linear/exponential upward slope to `joint_temperature`.
    \"\"\"
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_LF'
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        actual_current_col = feature_map['actual_current'][self.joint_index]
        
        # Temp column might not exist, but if it does, we map it
        temp_cols = feature_map.get('joint_temperature', [])
        temp_col = temp_cols[self.joint_index] if len(temp_cols) > self.joint_index else None
        
        mask = (df.index >= self.start_idx) & (df.index <= self.end_idx)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        # Severity increases linearly
        severity = np.linspace(0, 1.0, window_size)
        
        # 1. Current increase (friction requires more torque/current)
        # We increase current by up to 30% at max severity
        current_multiplier = 1.0 + (0.30 * severity)
        df.loc[mask, actual_current_col] = df.loc[mask, actual_current_col] * current_multiplier
        
        # 2. Temperature rise
        if temp_col and temp_col in df.columns:
            # Temperature increases steadily by up to 15 degrees C
            temp_increase = 15.0 * severity
            df.loc[mask, temp_col] = df.loc[mask, temp_col] + temp_increase
            
        full_severity = pd.Series(0.0, index=df.index)
        full_severity.loc[mask] = severity
        
        return self._apply_labels(df, self.fault_name, mask, full_severity)
