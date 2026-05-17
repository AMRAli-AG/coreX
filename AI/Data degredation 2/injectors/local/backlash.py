import pandas as pd
import numpy as np
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class BacklashInjector(BaseInjector):
    """
    Scenario A: Gear Backlash & Surface Wear (FAULT_BL)
    Physics-Informed: Non-linear Dead-Zone (2b) switching logic.
    Mechanisms: Kinematic Phase Lag, Zero-Crossing Spikes, and Re-engagement transients.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int):
        super().__init__(start_idx, end_idx, joint_index)
        self.fault_name = 'FAULT_BL'
        self.has_printed = False
        # Dead-zone expansion coefficient (beta)
        self.beta_backlash = 0.15 
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        df = df.copy()
        overlap_start, overlap_end = self.get_chunk_overlap(df)
        if overlap_start is None:
            return df
            
        if not self.has_printed:
            print(f"-> [MECHANICAL] Injecting {self.fault_name}: Non-linear Dead-Zone switching on Joint {self.joint_index}...")
            self.has_printed = True
            
        # Target channels
        pos_target_col = feature_map.get('target_q', [])[self.joint_index]
        pos_actual_col = feature_map.get('actual_q', [])[self.joint_index]
        vel_actual_col = feature_map.get('actual_qd', [])[self.joint_index]
        cur_actual_col = feature_map.get('actual_current', [])[self.joint_index]
        
        mask = (df.index >= overlap_start) & (df.index < overlap_end)
        duration = mask.sum()
        if duration == 0:
            return df
            
        severity = Labeler.calculate_severity(overlap_start, overlap_end, self.start_idx, self.end_idx, profile='exponential')
        
        # 1. Non-linear Dead-Zone Switching Logic
        if vel_actual_col and vel_actual_col in df.columns:
            vel_vals = df.loc[mask, vel_actual_col].values
            signs = np.sign(vel_vals)
            # Find zero-crossings (sign changes)
            # We use a small epsilon for zero-crossing detection in real telemetry
            sign_changes = np.concatenate([[False], np.diff(signs) != 0])
            
            # Cumulative fatigue effect: beta * number of reversals
            reversal_count = np.cumsum(sign_changes)
            dynamic_severity = severity * (1 + self.beta_backlash * reversal_count / max(1, reversal_count[-1]))
            
            # 2. Kinematic Phase Lag & Position Error Spike (Delta_q)
            if pos_actual_col and pos_actual_col in df.columns:
                pos_vals = df.loc[mask, pos_actual_col].values
                # Base dead-zone gap: 0.005 rad nominal
                gap_magnitude = 0.006 
                
                # Apply switching lag
                lag_signal = np.zeros_like(pos_vals)
                current_lag = 0.0
                for i in range(len(pos_vals)):
                    if sign_changes[i]:
                        # The 'Spike' occurs exactly at the reversal boundary
                        # Modeling the traversing of the dead-zone
                        current_lag = signs[i] * gap_magnitude
                    lag_signal[i] = current_lag
                
                df.loc[mask, pos_actual_col] = pos_vals - (lag_signal * dynamic_severity)
                
            # 3. Transient Current Re-engagement Strike
            if cur_actual_col and cur_actual_col in df.columns:
                # Create a mask for the strike window (3-5 samples after reversal)
                full_strike_mask = np.zeros(len(df), dtype=bool)
                indices = np.where(sign_changes)[0]
                for idx in indices:
                    window = np.arange(idx + 1, min(idx + 5, duration))
                    full_strike_mask[np.where(mask)[0][window]] = True
                
                # Align multipliers with the full dataframe
                full_multipliers = np.ones(len(df))
                sev_full = np.ones(len(df))
                sev_full[mask] = dynamic_severity
                full_multipliers[full_strike_mask] = 1.0 + (1.8 * sev_full[full_strike_mask])
                
                df.loc[full_strike_mask, cur_actual_col] *= full_multipliers[full_strike_mask]

                
        # 4. Cartesian TCP Repeatability Degradation
        tcp_cols = [c for c in df.columns if 'tcp' in c.lower() and ('pose' in c.lower() or 'actual' in c.lower())]
        if tcp_cols:
            # Inject spatial jitter: 0.02mm -> 0.6mm
            for col in tcp_cols:
                # 0.0006 m = 0.6 mm
                noise = (np.random.normal(0, 0.0006, duration)) * severity
                df.loc[mask, col] += noise

        return Labeler.apply_labels(df, overlap_start, overlap_end, self.fault_name, severity)
