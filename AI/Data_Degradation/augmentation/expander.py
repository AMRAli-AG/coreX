import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import warnings
from typing import List, Optional, Union
from scipy.interpolate import interp1d

# Advanced Augmentation Libraries
try:
    from tsaug import TimeWarp
    from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

class DataExpander:
    """
    Advanced Physics-Informed Data Expander for Robotic Sensor Data.
    Implements TimeGAN, TimeWarp, and Endpoint Smoothing for continuous signals.
    """
    def __init__(self, filepath: str, window_size: int = 24, smoothing_window: int = 10):
        self.filepath = filepath
        self.window_size = window_size
        self.smoothing_window = smoothing_window  # Rows to use for transition
        self.df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        
        if not HAS_ADVANCED:
            warnings.warn("Advanced libraries (tsaug, ydata-synthetic) not found. Falling back to basic expansion.")

    def load_data(self) -> pd.DataFrame:
        """Loads dataset and identifies feature columns for augmentation."""
        print(f"[*] Loading dataset from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        # Identify features (exclude time and labels)
        self.features = [col for col in self.df.columns if 'time' not in col.lower() and 'label' not in col.lower() and col.lower() != 't']
        return self.df

    def _prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Converts flat data to sequences of (samples, window_size, features)."""
        X = [data[i : i + self.window_size] for i in range(len(data) - self.window_size + 1)]
        return np.array(X)

    def _apply_boundary_smoothing(self, df_prev: pd.DataFrame, df_next: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures physics-informed continuity at the concatenation boundary.
        Uses a linear cross-fade/interpolation to eliminate sharp jumps.
        """
        if df_prev is None or df_next is None:
            return df_next
            
        N = self.smoothing_window
        if len(df_prev) < N or len(df_next) < N:
            return df_next
            
        df_next_smooth = df_next.copy()
        
        # Endpoint Matching for each feature
        for feature in self.features:
            if feature not in df_prev.columns or feature not in df_next.columns:
                continue
                
            target_start = df_prev[feature].iloc[-1]
            original_values = df_next[feature].values[:N]
            
            # weights: 1.0 (target_start) -> 0.0 (original_values[N-1])
            weights = np.linspace(1.0, 0.0, N)
            
            # CRITICAL FIX: Use position-based indexing (iloc) to prevent index mismatch crashes
            smoothed_values = (weights * target_start) + ((1 - weights) * original_values)
            col_idx = df_next_smooth.columns.get_loc(feature)
            df_next_smooth.iloc[0:N, col_idx] = smoothed_values
            
        return df_next_smooth

    def _reconstruct_dataframe(self, sequences: np.ndarray, original_df: pd.DataFrame) -> pd.DataFrame:
        """Converts augmented sequences back to a continuous DataFrame."""
        # Use first step of each window, plus the remaining tail of the last window
        flattened = sequences[:, 0, :]
        last_window_tail = sequences[-1, 1:, :]
        final_array = np.vstack([flattened, last_window_tail])
        
        new_df = pd.DataFrame(final_array, columns=self.features)
        
        # Reconstruct Time (Strict Continuity)
        time_cols = [col for col in original_df.columns if 'time' in col.lower() or col.lower() == 't']
        if time_cols:
            time_col = time_cols[0]
            time_step = original_df[time_col].iloc[1] - original_df[time_col].iloc[0] if len(original_df) > 1 else 0.01
            start_time = original_df[time_col].max() + time_step
            new_df[time_col] = start_time + np.arange(len(new_df)) * time_step
        
        # Initialize Labels
        if 'fault_label' in original_df.columns:
            new_df['fault_label'] = 'Healthy'
        if 'severity_score' in original_df.columns:
            new_df['severity_score'] = 0.0
            
        return new_df

    def augment_statistical(self, factor: int = 2) -> pd.DataFrame:
        """Applies TimeWarp augmentation using tsaug."""
        if not HAS_ADVANCED: raise ImportError("tsaug required for statistical augmentation.")
        if self.df is None: self.load_data()
        
        data_values = self.df[self.features].values
        seqs = self._prepare_sequences(data_values)
        
        print("[+] Initializing TimeWarp transformation...")
        augmenter = TimeWarp() * (factor - 1)
        augmented_seqs = augmenter.augment(seqs)
        
        return self._reconstruct_dataframe(augmented_seqs, self.df)

    def augment_generative(self, factor: int = 2, epochs: int = 50) -> pd.DataFrame:
        """Trains TimeGAN to generate synthetic sequences."""
        if not HAS_ADVANCED: raise ImportError("ydata-synthetic required for generative augmentation.")
        if self.df is None: self.load_data()
        
        data_values = self.df[self.features].values
        seqs = self._prepare_sequences(data_values)
        
        gan_args = ModelParameters(batch_size=128, lr=5e-4, noise_dim=32, layers_dim=128)
        train_args = TrainParameters(epochs=epochs, sequence_length=self.window_size, number_sequences=len(seqs))
        
        print(f"[+] Training TimeGAN Synthesizer for {epochs} epochs...")
        synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
        # Note: Synthesizer internal progress bar usually handles the loop
        synth.fit(seqs, train_args)
        
        print(f"[+] Sampling synthetic data (Factor: {factor})...")
        synthetic_seqs = synth.sample(len(seqs) * (factor - 1))
        return self._reconstruct_dataframe(synthetic_seqs, self.df)

    def expand(self, factor: int = 2, method: str = 'gan', epochs: int = 50) -> pd.DataFrame:
        """Main entry point for dataset expansion."""
        if self.df is None: self.load_data()
        if factor <= 1: return self.df
            
        print(f"[*] Starting {factor}x Expansion via '{method}'...")
        start_t = time.time()
        
        if method == 'gan' and HAS_ADVANCED:
            new_chunks = self.augment_generative(factor, epochs=epochs)
        elif method == 'warp' and HAS_ADVANCED:
            new_chunks = self.augment_statistical(factor)
        else:
            if method not in ['basic', 'none']: print(f"[!] Falling back to basic physics-informed expansion.")
            new_chunks = self._basic_expand_chunks(factor)
            
        # Apply Boundary Smoothing between original and new data
        print("[+] Performing final endpoint matching and boundary smoothing...")
        smoothed_new_data = self._apply_boundary_smoothing(self.df, new_chunks)
        
        final_df = pd.concat([self.df, smoothed_new_data], ignore_index=True)
        
        duration = time.time() - start_t
        print(f"[#] Expansion complete ({duration:.2f}s)! {len(self.df)} -> {len(final_df)} rows.")
        return final_df

    def _basic_expand_chunks(self, factor: int) -> pd.DataFrame:
        """Helper for basic physics-informed duplication with continuity."""
        chunks = []
        last_df = self.df
        print(f"[+] Concatenating {factor-1} segments with boundary smoothing...")
        for _ in tqdm(range(1, factor), desc="Stitching Chunks"):
            df_copy = self.df.copy()
            
            # Time Continuity
            time_cols = [col for col in df_copy.columns if 'time' in col.lower() or col.lower() == 't']
            if time_cols:
                col = time_cols[0]
                prev_max = last_df[col].max()
                time_step = self.df[col].iloc[1] - self.df[col].iloc[0] if len(self.df) > 1 else 0.01
                df_copy[col] = df_copy[col] + (prev_max - self.df[col].min()) + time_step
            
            df_copy = self._apply_boundary_smoothing(last_df, df_copy)
            chunks.append(df_copy)
            last_df = df_copy
            
        return pd.concat(chunks, ignore_index=True)
