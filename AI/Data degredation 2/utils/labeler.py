import pandas as pd
import numpy as np

class Labeler:
    """
    Centralized utility for fault labeling and dynamic severity calculation.
    Ensures a DRY architecture across all fault injectors.
    """
    @staticmethod
    def initialize_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Initializes fault_label and severity_score if they don't exist."""
        if 'fault_label' not in df.columns:
            df['fault_label'] = 'Healthy'
        if 'severity_score' not in df.columns:
            df['severity_score'] = 0.0
        return df

    @staticmethod
    def calculate_severity(overlap_start: int, overlap_end: int, total_start: int, total_end: int, profile: str = 'exponential') -> np.ndarray:
        """
        Calculates a severity array for the current chunk overlap.
        Profiles: 'exponential' (incipient onset), 'linear' (steady wear), 'constant' (step change).
        """
        duration = overlap_end - overlap_start
        total_duration = total_end - total_start
        relative_start = overlap_start - total_start
        
        # Normalized time progress (0.0 to 1.0)
        t = np.linspace(relative_start/total_duration, (relative_start + duration)/total_duration, duration, endpoint=False)
        
        if profile == 'exponential':
            severity = np.exp(t * 4.6) / 100.0
        elif profile == 'linear':
            severity = t
        elif profile == 'polynomial':
            severity = t ** 2
        elif profile == 'abrupt':
            severity = (t > 0.15).astype(float)
        elif profile == 'constant':
            severity = np.ones(duration)
        else:
            raise ValueError(f"Unknown severity profile: {profile}")

        return np.clip(severity, 0, 1.0)

    @staticmethod
    def apply_labels(df: pd.DataFrame, start_idx: int, end_idx: int, fault_name: str, severity_array: np.ndarray) -> pd.DataFrame:
        """Applies labels and scores to the designated slice and logs to terminal."""
        Labeler.initialize_labels(df)
        end_idx = min(end_idx, len(df))
        duration = end_idx - start_idx
        
        if duration <= 0:
            return df
            
        df.loc[start_idx:end_idx-1, 'fault_label'] = fault_name
        df.loc[start_idx:end_idx-1, 'severity_score'] = severity_array[:duration]
        
        # Real-time streaming log for terminal UI
        max_sev = np.max(severity_array[:duration])
        print(f"   [Labeler] {fault_name} Active | Target Range: {start_idx}:{end_idx} | Current Max Severity: {max_sev:.4f}")
        
        return df
