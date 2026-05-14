import pandas as pd
import numpy as np
from ..base_injector import BaseInjector

class PayloadChangeInjector(BaseInjector):
    \"\"\"
    Scenario D: Payload/Propagation (FAULT_SYS)
    
    Clinical Description:
    An unexpected increase in the end-effector payload (e.g., picking up a heavier object 
    or a collision) affects the entire kinematic chain.
    Physically, this is a system-wide injection. To maintain the trajectory, all joints 
    must exert higher torque, drawing more current. Additionally, structural deflection 
    increases, reducing the accuracy of the Tool Center Point (TCP) pose.
    
    Mathematical Logic:
    1. Apply a proportional increase to `actual_current` across all joints.
    2. Add an error offset to end-effector/TCP pose features (if they exist).
    \"\"\"
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int = None):
        # Joint index is ignored as this is system-wide, but kept for interface consistency
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_SYS'
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        all_current_cols = feature_map.get('actual_current', [])
        
        mask = (df.index >= self.start_idx) & (df.index <= self.end_idx)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        # Payload change is often sudden, so severity might just step up
        # We will use a step function (sudden 50% increase in payload equivalent)
        severity = np.ones(window_size) * 1.0
        
        current_multiplier = 1.5  # 50% increase
        
        for col in all_current_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] * current_multiplier
                
        # Optional: TCP offset logic if tcp columns exist
        tcp_cols = feature_map.get('tcp_pose', [])
        for col in tcp_cols:
            if col in df.columns:
                df.loc[mask, col] += np.random.normal(0, 0.05, window_size)
                
        full_severity = pd.Series(0.0, index=df.index)
        full_severity.loc[mask] = severity
        
        return self._apply_labels(df, self.fault_name, mask, full_severity)
