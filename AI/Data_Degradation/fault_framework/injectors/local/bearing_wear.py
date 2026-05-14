import pandas as pd
import numpy as np
from ..base_injector import BaseInjector

class BearingWearInjector(BaseInjector):
    \"\"\"
    Scenario C: Bearing Wear (FAULT_BW)
    
    Clinical Description:
    Spalling or pitting on bearing races introduces high-frequency micro-vibrations 
    into the mechanical system.
    Physically, these vibrations manifest in the velocity (`actual_qd`) and the motor 
    current (`actual_current`) as stochastic noise. As wear progresses, the RMS 
    (Root Mean Square) of the noise increases, and the probability of sharp extreme 
    peaks (Kurtosis) increases significantly.
    
    Mathematical Logic:
    1. Inject high-frequency stochastic noise into `actual_qd` and `actual_current`.
    2. Scale the amplitude of the noise (RMS) over time.
    3. Introduce sporadic high peaks to artificially raise the signal Kurtosis.
    \"\"\"
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BW'
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        
        actual_qd_col = feature_map['actual_qd'][self.joint_index]
        actual_current_col = feature_map['actual_current'][self.joint_index]
        
        mask = (df.index >= self.start_idx) & (df.index <= self.end_idx)
        window_size = mask.sum()
        
        if window_size == 0:
            return df
            
        severity = np.linspace(0.01, 1.0, window_size)
        
        # Baseline noise scales with current velocity/current magnitude
        base_qd = df.loc[mask, actual_qd_col].abs().mean()
        base_curr = df.loc[mask, actual_current_col].abs().mean()
        
        # Inject standard Gaussian noise (increasing RMS)
        qd_noise = np.random.normal(0, base_qd * 0.1, window_size) * severity
        curr_noise = np.random.normal(0, base_curr * 0.1, window_size) * severity
        
        # Inject sparse, high-amplitude peaks to increase Kurtosis
        # Probability of a peak increases with severity
        peak_prob = 0.05 * severity
        peak_mask = np.random.random(window_size) < peak_prob
        
        qd_peaks = peak_mask * np.random.normal(0, base_qd * 0.5, window_size)
        curr_peaks = peak_mask * np.random.normal(0, base_curr * 0.5, window_size)
        
        df.loc[mask, actual_qd_col] += (qd_noise + qd_peaks)
        df.loc[mask, actual_current_col] += (curr_noise + curr_peaks)
        
        full_severity = pd.Series(0.0, index=df.index)
        full_severity.loc[mask] = severity
        
        return self._apply_labels(df, self.fault_name, mask, full_severity)
