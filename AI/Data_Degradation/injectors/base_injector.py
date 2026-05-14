from abc import ABC, abstractmethod
import pandas as pd
from utils.labeler import Labeler

class BaseInjector(ABC):
    """
    Abstract Base Class for all Fault Injectors.
    Provides the standard interface to ensure modularity.
    """
    
    def __init__(self, start_idx: int, end_idx: int, joint_index: int = None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.joint_index = joint_index
        self.fault_name = "BaseFault"

    @abstractmethod
    def inject(self, df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
        pass

    def get_chunk_overlap(self, df: pd.DataFrame):
        """Returns the start and end indices within the current dataframe chunk."""
        chunk_start = df.index.min()
        chunk_end = df.index.max()
        
        overlap_start = max(chunk_start, self.start_idx)
        overlap_end = min(chunk_end + 1, self.end_idx)
        
        if overlap_start < overlap_end:
            return overlap_start, overlap_end
        return None, None
