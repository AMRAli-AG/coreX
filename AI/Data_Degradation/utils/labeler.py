import pandas as pd
import numpy as np

class Labeler:
    @staticmethod
    def initialize_labels(df):
        """Initializes fault_label and severity_score if they don't exist."""
        if 'fault_label' not in df.columns:
            df['fault_label'] = 'Healthy'
        if 'severity_score' not in df.columns:
            df['severity_score'] = 0.0
        return df

    @staticmethod
    def apply_labels(df, start_idx, end_idx, fault_name, severity_array):
        """
        Applies the fault label and precise severity score to the designated slice.
        
        Args:
            df: The main pandas DataFrame.
            start_idx: The starting row index for the fault.
            end_idx: The ending row index for the fault.
            fault_name: String label for the fault (e.g., 'FAULT_BL').
            severity_array: An array/series of severity scores (0.0 to 1.0) matching the slice length.
        """
        Labeler.initialize_labels(df)
        
        # Ensure we don't go out of bounds
        end_idx = min(end_idx, len(df))
        duration = end_idx - start_idx
        
        if duration <= 0:
            return df
            
        # Update labels
        df.loc[start_idx:end_idx-1, 'fault_label'] = fault_name
        
        # If there's an existing severity score (e.g. from a previous fault in a Combo),
        # we can take the maximum or add them, but standard practice is to use the max
        # or just override if it's a new primary fault. Here we override.
        df.loc[start_idx:end_idx-1, 'severity_score'] = severity_array[:duration]
        
        return df
