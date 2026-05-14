import pandas as pd
import numpy as np
from tqdm import tqdm
import time

class DataExpander:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        print(f"Loading dataset from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        return self.df

    def expand(self, factor=2):
        if self.df is None:
            self.load_data()
            
        if factor <= 1:
            print("Factor must be > 1 to expand. Returning original dataset.")
            return self.df
            
        print(f"Expanding dataset by {factor}x...")
        original_rows = len(self.df)
        
        # We will concatenate the dataframe multiple times
        expanded_dfs = [self.df]
        
        for i in tqdm(range(1, factor), desc="Concatenating Data"):
            df_copy = self.df.copy()
            # If there is a 'time' or 'timestamp' column, we should make it continuous
            time_cols = [col for col in df_copy.columns if 'time' in col.lower() or col.lower() == 't']
            for col in time_cols:
                max_time = expanded_dfs[-1][col].max()
                time_step = self.df[col].iloc[1] - self.df[col].iloc[0] if len(self.df) > 1 else 0.01
                df_copy[col] = df_copy[col] + max_time + time_step
                
            expanded_dfs.append(df_copy)
            
        print("Finalizing expanded dataset...")
        final_df = pd.concat(expanded_dfs, ignore_index=True)
        print(f"Expansion complete! Original rows: {original_rows} -> New rows: {len(final_df)}")
        return final_df
