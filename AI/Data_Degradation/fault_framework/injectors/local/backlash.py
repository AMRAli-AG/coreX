import pandas as pd
import numpy as np
from ..base_injector import BaseInjector

class BacklashInjector(BaseInjector):
    \"\"\"
    Scenario A: Gear Backlash (FAULT_BL)
    
    Clinical Description:
    Mechanical backlash occurs when there is clearance or play between mating gear teeth.
    Physically, this causes a time-lag and positional error (hysteresis) specifically 
    when the motor changes direction (velocity sign flip). Over time, as wear increases, 
    the play increases, meaning the positional lag grows exponentially.
    
    Mathematical Logic:
    1. Identify direction changes in the target trajectory (`target_qd` crosses zero).
    2. During these changes, apply a lag to `actual_q`.
    3. The magnitude of this lag increases exponentially over the injection window.
    \"\"\"
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BL'
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        # Identify columns
        target_q_col = feature_map['target_q'][self.joint_index]
        actual_q_col = feature_map['actual_q'][self.joint_index]
        target_qd_col = feature_map.get('target_qd', [None] * (self.joint_index+1))[self.joint_index]
        
        # Calculate velocity if target_qd is not available
        if target_qd_col is None or target_qd_col not in df.columns:
            target_qd = df[target_q_col].diff().fillna(0)
        else:
            target_qd = df[target_qd_col]
            
        # Create mask for the injection window
        mask = (df.index >= self.start_idx) & (df.index <= self.end_idx)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        # Severity increases exponentially from 0.01 to 1.0
        time_steps = np.linspace(0, 1, window_size)
        severity = np.exp(time_steps * 4.6) / 100.0  # Approx 0.01 to 1.0
        
        # Identify direction changes (velocity sign flips)
        sign_flips = np.sign(target_qd).diff().ne(0) & (target_qd != 0)
        
        # Apply lag only during direction changes, scaled by severity
        # Max backlash positional error assumed to be 0.05 radians
        max_backlash = 0.05 
        
        lag_effect = np.zeros(len(df))
        lag_effect[mask] = sign_flips[mask] * max_backlash * severity
        
        # Smear the effect slightly to make it realistic (moving average)
        lag_effect = pd.Series(lag_effect).rolling(window=5, min_periods=1).mean()
        
        # Modify actual_q
        df.loc[mask, actual_q_col] = df.loc[mask, actual_q_col] - lag_effect[mask] * np.sign(target_qd[mask])
        
        # Construct full severity series for labeling
        full_severity = pd.Series(0.0, index=df.index)
        full_severity.loc[mask] = severity
        
        return self._apply_labels(df, self.fault_name, mask, full_severity)
