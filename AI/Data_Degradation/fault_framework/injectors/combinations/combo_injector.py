import pandas as pd
from ..base_injector import BaseInjector

class ComboInjector(BaseInjector):
    \"\"\"
    Handles combinations of multiple faults (e.g., Backlash + Bearing Wear).
    Executes multiple injectors sequentially on the same data chunk.
    \"\"\"
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int, injectors: list):
        super().__init__(start_idx, end_idx, joint_index)
        self.injectors = injectors
        # Create a combined name
        self.fault_name = "_PLUS_".join([inj.fault_name for inj in injectors])
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        result_df = df.copy()
        for inj in self.injectors:
            # Override indices to ensure alignment
            inj.start_idx = self.start_idx
            inj.end_idx = self.end_idx
            if self.joint_index is not None:
                inj.joint_index = self.joint_index
                
            result_df = inj.inject(result_df, feature_map)
            
        # Note: The base injectors will overwrite fault_label. 
        # We enforce the combo name at the end.
        mask = (result_df.index >= self.start_idx) & (result_df.index <= self.end_idx)
        if 'fault_label' in result_df.columns:
            result_df.loc[mask, 'fault_label'] = self.fault_name
            
        return result_df
