from abc import ABC, abstractmethod
import pandas as pd

class BaseInjector(ABC):
    \"\"\"
    Abstract Base Class for all Fault Injectors.
    Provides the standard interface to ensure modularity.
    \"\"\"
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int = None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.joint_index = joint_index

    @abstractmethod
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        \"\"\"
        Injects the specific fault into the provided dataframe.
        
        Args:
            df (pd.DataFrame): The data chunk to modify.
            feature_map (dict): A mapping dictionary (e.g., {'q': ['joint1_q', ...], 'current': [...]})
                                to dynamically find relevant columns.
                                
        Returns:
            pd.DataFrame: The modified dataframe with new columns: 'fault_label' and 'severity_score'.
        \"\"\"
        pass

    def _apply_labels(self, df: pd.DataFrame, fault_name: str, mask: pd.Series, severity: pd.Series):
        \"\"\"Helper to standardize labeling.\"\"\"
        if 'fault_label' not in df.columns:
            df['fault_label'] = 'Normal'
        if 'severity_score' not in df.columns:
            df['severity_score'] = 0.0

        # Apply labels only where mask is True
        df.loc[mask, 'fault_label'] = fault_name
        df.loc[mask, 'severity_score'] = severity[mask]
        return df
