import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class LubricationFailureInjector(BaseInjector):
    """
    Scenario LF: Distributed Lubrication Degradation & Viscous Friction Increase.
    Research Validation: IEEE Standard for PHM in Robotics (Friction Models).
    
    Mathematical Model:
    - τ_fric = B * q_dot + C * sign(q_dot)
    - Fault: B_degraded = B_nominal * (1 + k_fric * Severity)
    - Thermal Coupling: ΔT ∝ P_fric = τ_fric * q_dot
    """
    
    def __init__(self, start_idx: int, end_idx: int):
        super().__init__(start_idx, end_idx)
        self.fault_name = 'FAULT_LF'
        self.viscous_factor = 2.8      # Coefficient for friction increase
        self.thermal_coupling = 0.12   # Degrees C per sample (at max severity)
        self.is_distributed = True

    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # Calculate severity profile (Linear growth for lubrication)
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='linear')
        
        # Lubrication failure is system-wide (Distributed across all joints)
        for j in range(6):
            cur_col = feature_map.get('actual_current', [])[j]
            vel_col = feature_map.get('actual_qd', [])[j]
            temp_col = feature_map.get('joint_temperature', [])[j]
            
            # 1. Viscous Drag -> Proportional Current Elevation
            if cur_col in df.columns and vel_col in df.columns:
                vel_abs = np.abs(df.loc[mask, vel_col].values)
                # I_degraded = I_baseline + k * |v| * severity
                drag_current = vel_abs * self.viscous_factor * severity
                df.loc[mask, cur_col] += drag_current
                
            # 2. Thermal Runaway (Thermodynamic Gradient)
            if temp_col in df.columns:
                # Cumulative heat build-up
                time_range = np.arange(duration)
                temp_rise = time_range * self.thermal_coupling * (severity.max() if isinstance(severity, np.ndarray) else severity)
                df.loc[mask, temp_col] += temp_rise
                
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
