import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class ElectricalErosionInjector(BaseInjector):
    """
    Scenario D: Electrical Erosion & EDM Discharge (FAULT_EE)
    Physics-Informed: Parasitic currents and Bearing Fluting.
    Signature: Wideband Current 'Hiss', Arcing Spikes, and Step-wise Thermal Jumps.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_EE'
        self.has_printed = False
        # Accelerated degradation step (delta)
        self.delta_erosion = 4.5 
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> [ELECTRO-MECHANICAL] Injecting {self.fault_name}: EDM Bearing Discharge & Fluting on Joint {self.joint_index}...")
            self.has_printed = True
            
        # Target channels
        cur_actual_col = feature_map.get('actual_current', [])[self.joint_index]
        temp_col = feature_map.get('joint_temperature', [])[self.joint_index]
        vel_actual_col = feature_map.get('actual_qd', [])[self.joint_index]
        
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # Accelerated Step-wise Progression
        # Unlike linear wear, EE progresses in abrupt jumps as arcs breach the dielectric
        t_chunk = np.linspace(0, 1, duration)
        raw_severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='linear')
        # Step-wise transformation
        severity = np.floor(raw_severity * 5) / 5 # 5 discrete degradation steps
        
        # 1. Electrical Hiss Noise & Arcing Spikes
        if cur_actual_col and cur_actual_col in df.columns:
            cur_vals = df.loc[mask, cur_actual_col].values
            # Continuous wideband Hiss noise (VFD Switching signature)
            hiss_noise = np.random.normal(0, 0.15 * severity, duration)
            
            # Stochastic Arcing Impulse Spikes
            arc_spikes = np.zeros(duration)
            # 2% probability of a macroscopic arc event per sample
            spike_indices = np.where(np.random.rand(duration) < 0.02)[0]
            arc_spikes[spike_indices] = (np.random.rand(len(spike_indices)) * 2.5) * severity[spike_indices]

            
            df.loc[mask, cur_actual_col] = cur_vals + hiss_noise + arc_spikes
            
        # 2. Sudden Frictional Thermal Jump (Step-wise)
        if temp_col and temp_col in df.columns:
            temp_vals = df.loc[mask, temp_col].values
            # Sudden step-climb modeling fluted ridge friction
            thermal_step = 8.0 * severity
            df.loc[mask, temp_col] = temp_vals + thermal_step
            
        # 3. Kinematic Velocity Instability (Torque Ripple)
        if vel_actual_col and vel_actual_col in df.columns:
            vel_vals = df.loc[mask, vel_actual_col].values
            # Low-frequency hunting/roughness induced by fluting
            t_axis = np.arange(duration) * 0.008
            hunting_ripple = 0.05 * severity * np.sin(2 * np.pi * 5 * t_axis)
            df.loc[mask, vel_actual_col] = vel_vals + hunting_ripple
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
