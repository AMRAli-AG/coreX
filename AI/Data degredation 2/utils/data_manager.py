import pandas as pd
import re

class DataManager:
    """
    Handles data loading, chunking, and automatic feature mapping for high-dimensional robotic telemetry.
    
    Industry 4.0 Feature Space:
    - Profiles 69 physical attributes across 6 joints.
    - Classifies features into Electrical/Dynamic, Thermal, and Kinematic/Control domains.
    - Enables Residual Gap Analysis (Target vs. Actual) for anomaly detection.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.columns = pd.read_csv(filepath, nrows=0).columns.tolist()
        self.feature_map = self._auto_map_features(self.columns)
        
    def _auto_map_features(self, columns: list) -> dict:
        """
        Dynamically maps column names to standard feature types for each joint.
        Assumes standard naming like 'joint1_q', 'j0_current', 'actual_q_2', etc.
        """
        feature_map = {
            'target_q': {},
            'actual_q': {},
            'target_qd': {},
            'actual_qd': {},
            'actual_current': {},
            'target_moment': {},
            'joint_temperature': {}
        }
        
        for col in columns:
            col_lower = col.lower()
            
            # Extract joint index (look for numbers)
            match = re.search(r'\d+', col_lower)
            if not match:
                continue
            j_idx = int(match.group())
            
            # Categorize
            if 'temp' in col_lower:
                feature_map['joint_temperature'][j_idx] = col
            elif 'current' in col_lower or 'i_actual' in col_lower:
                feature_map['actual_current'][j_idx] = col
            elif 'moment' in col_lower or 'tau' in col_lower:
                feature_map['target_moment'][j_idx] = col
            elif 'target' in col_lower and 'qd' in col_lower:
                feature_map['target_qd'][j_idx] = col
            elif 'target' in col_lower and 'q' in col_lower:
                feature_map['target_q'][j_idx] = col
            elif 'qd' in col_lower or 'vel' in col_lower:
                feature_map['actual_qd'][j_idx] = col
            elif 'q' in col_lower or 'pos' in col_lower:
                feature_map['actual_q'][j_idx] = col
                
        # Convert dict of dicts to list based on max joint index for easier access
        max_j = 0
        for f in feature_map:
            if feature_map[f]:
                max_j = max(max_j, max(feature_map[f].keys()))
                
        list_map = {}
        for f in feature_map:
            list_map[f] = [feature_map[f].get(i, None) for i in range(max_j + 1)]
            
        return list_map
        
    def process_and_save(self, injector, output_path: str, chunksize: int = 100000):
        """
        Reads the CSV in chunks, applies the injector, and saves to output_path.
        """
        print(f"Processing data in chunks of {chunksize}...")
        first_chunk = True
        
        # We need to keep track of absolute index across chunks
        current_idx = 0
        
        for chunk in pd.read_csv(self.filepath, chunksize=chunksize):
            chunk.index = range(current_idx, current_idx + len(chunk))
            
            # Inject fault
            modified_chunk = injector.inject(chunk, self.feature_map)
            
            # Save
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            modified_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            first_chunk = False
            current_idx += len(chunk)
            
        print(f"Processing complete. Saved to {output_path}")
