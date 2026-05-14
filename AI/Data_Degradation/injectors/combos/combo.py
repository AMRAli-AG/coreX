import pandas as pd
from injectors.base_injector import BaseInjector
from utils.labeler import Labeler

class ComboInjector(BaseInjector):
    """
    Handles combinations of multiple faults (e.g., Backlash + Bearing Wear).
    Executes multiple injectors sequentially on the same data chunk.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int, injectors: list):
        super().__init__(start_idx, end_idx, joint_index)
        self.injectors = injectors
        self.fault_name = "_PLUS_".join([inj.fault_name for inj in injectors])
        self.has_printed = False
        
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        result_df = df.copy()
        
        overlap_start, overlap_end = self.get_chunk_overlap(result_df)
        if overlap_start is None:
            return result_df
            
        if not self.has_printed:
            print(f"-> Injecting COMBO {self.fault_name} on Joint {self.joint_index}...")
            self.has_printed = True
            
        for inj in self.injectors:
            # Override indices to ensure alignment
            inj.start_idx = self.start_idx
            inj.end_idx = self.end_idx
            if self.joint_index is not None:
                inj.joint_index = self.joint_index
                
            result_df = inj.inject(result_df, feature_map)
            
        # The severity score will be roughly the max or latest from the sub-injectors
        # We just need to correct the label name
        mask = (result_df.index >= overlap_start) & (result_df.index < overlap_end)
        if 'fault_label' in result_df.columns:
            result_df.loc[mask, 'fault_label'] = self.fault_name
            
        return result_df
