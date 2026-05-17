import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class LubricationFailureInjector(BaseInjector):
    """
    Scenario B: Lubrication Failure (FAULT_LF)
    Tribological Profile: Transition from Hydrodynamic to Boundary Lubrication.
    Physics: Simultaneous modulation of Electrical (Current), Kinematic (Lag), and Thermal (Ramp).
    Onset: Incipient (Slow-onset exponential degradation).
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int = None):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_LF'
        self.has_printed = False
        self.temp_baselines = {0: 26.5, 1: 26.5, 2: 26.5, 3: 35.1, 4: 35.1, 5: 35.1}
        # Incipient degradation factor (alpha)
        self.alpha_lubrication = 3.5 
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            scope = f"Joint {self.joint_index}" if self.joint_index is not None else "All Joints (Distributed)"
            print(f"-> [TRIBOLOGY] Injecting {self.fault_name}: Boundary Lubrication Transition on {scope}...")
            self.has_printed = True
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # Incipient Exponential Severity Profile
        # t goes from 0 to 1 over the full fault duration
        # s(t) = (exp(alpha * t) - 1) / (exp(alpha) - 1)
        t_global = np.linspace(0, 1, self.end_idx - self.start_idx)
        # Map global t to current chunk
        chunk_t_start = (overlap_start - self.start_idx) / (self.end_idx - self.start_idx)
        chunk_t_end = (overlap_end - self.start_idx) / (self.end_idx - self.start_idx)
        t_chunk = np.linspace(chunk_t_start, chunk_t_end, duration)
        
        severity = (np.exp(self.alpha_lubrication * t_chunk) - 1) / (np.exp(self.alpha_lubrication) - 1)
        
        cur_cols = feature_map.get('actual_current', [])
        temp_cols = feature_map.get('joint_temperature', [])
        pos_cols = feature_map.get('actual_q', [])
        
        # Determine target joints
        target_joints = [self.joint_index] if self.joint_index is not None else range(6)
        joint_weights = {0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.5}
        
        for j_idx in target_joints:
            weight = joint_weights.get(j_idx, 1.0)
            local_severity = severity * weight
            
            # 1. Electrical Domain: Continuous RMS Current Increase
            if j_idx < len(cur_cols) and cur_cols[j_idx] in df.columns:
                col = cur_cols[j_idx]
                df.loc[mask, col] *= (1 + 0.45 * local_severity)
                
            # 2. Kinematic Domain: Progressive Tracking Lag (Delta_q)
            # Frictional counter-torque causes the motor to lag behind target
            if j_idx < len(pos_cols) and pos_cols[j_idx] in df.columns:
                col = pos_cols[j_idx]
                # Induce lag: 0.005 rad max deviation
                lag_magnitude = 0.008 
                df.loc[mask, col] -= (lag_magnitude * local_severity)
                
            # 3. Thermodynamic Domain: Exponential Thermal Ramp (dT/dt)
            if j_idx < len(temp_cols) and temp_cols[j_idx] in df.columns:
                col = temp_cols[j_idx]
                baseline = self.temp_baselines.get(j_idx, 25.0)
                # Irreversible mechanical friction work -> Thermal energy
                # dT/dt is proportional to (current * local_severity)
                temp_rise = 15.0 * local_severity
                df.loc[mask, col] = np.maximum(df.loc[mask, col], baseline) + temp_rise
                
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
