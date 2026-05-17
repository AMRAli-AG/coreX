import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler
class BearingWearInjector(BaseInjector):
    """
    Scenario C: Bearing Wear & Rolling Contact Fatigue (RCF)
    Physics-Informed: Cyclic Shock Pulses and Spectral Sidebands.
    Signature: High-frequency Velocity Jitter and Current Sideband harmonics.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BW'
        self.has_printed = False
        # Progressive spalling maturation amplitude (gamma)
        self.gamma_bearing = 2.8 
        self.dt = 0.008 # 125Hz
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> [VIBRATION] Injecting {self.fault_name}: High-Frequency RCF Sidebands on Joint {self.joint_index}...")
            self.has_printed = True
            
        # Target channels
        vel_actual_col = feature_map.get('actual_qd', [])[self.joint_index]
        cur_actual_col = feature_map.get('actual_current', [])[self.joint_index]
        temp_col = feature_map.get('joint_temperature', [])[self.joint_index]
        
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        # Maturation Profile: Linear progression of spall energy
        t_chunk = np.linspace(0, 1, duration)
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='linear')
        
        # 1. Kinematic Velocity Jitter (High Kurtosis/Crest Factor)
        if vel_actual_col and vel_actual_col in df.columns:
            vel_vals = df.loc[mask, vel_actual_col].values
            # Stochastic Gaussian micro-oscillations
            # Magnitude increases with severity and gamma
            jitter_amplitude = 0.012 * severity * self.gamma_bearing
            jitter_noise = np.random.normal(0, jitter_amplitude, duration)
            
            # Add cyclic shock pulses (spalls)
            # Modeling rolling element impacts (e.g., every 15 samples)
            spall_pulses = np.zeros(duration)
            spall_indices = np.arange(0, duration, 12)
            spall_pulses[spall_indices] = jitter_amplitude[spall_indices] * 2.5

            
            df.loc[mask, vel_actual_col] = vel_vals + jitter_noise + spall_pulses
            
        # 2. Electrical Spectral Sidebands (MCSA Signature)
        if cur_actual_col and cur_actual_col in df.columns:
            cur_vals = df.loc[mask, cur_actual_col].values
            # Inject harmonic sidebands: w_fault = |fs +/- k * f_bearing|
            # We simulate this by superimposing low-amplitude sine components
            t_axis = np.arange(duration) * self.dt
            f_bearing = 18.5 # Hz (characteristic defect frequency)
            f_supply = 50.0  # Hz
            
            # Sideband 1: fs - f_bearing
            s1 = 0.08 * severity * np.sin(2 * np.pi * (f_supply - f_bearing) * t_axis)
            # Sideband 2: fs + f_bearing
            s2 = 0.08 * severity * np.sin(2 * np.pi * (f_supply + f_bearing) * t_axis)
            
            df.loc[mask, cur_actual_col] = cur_vals + s1 + s2
            
        # 3. Thermal Localized Energy Dissipation
        if temp_col and temp_col in df.columns:
            temp_vals = df.loc[mask, temp_col].values
            # Localized thermal offset due to cyclic micro-friction
            thermal_offset = 4.5 * severity
            df.loc[mask, temp_col] = temp_vals + thermal_offset
            
        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
