# -*- coding: utf-8 -*-
"""
Time Warping Data Augmentation Technique.
Dynamically warp the time axes of sequential telemetry.
Leverages 'tsaug' if available; otherwise falls back to high-fidelity scipy interpolation.
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Attempt import of advanced libraries
try:
    from tsaug import TimeWarp
    HAS_TSAUG = True
except ImportError:
    HAS_TSAUG = False
    from scipy.interpolate import interp1d

# Standard technique identification
TECHNIQUE_NAME = "Time Warping"


def _pure_numpy_time_warp(X: np.ndarray, num_anchors: int = 4, sigma: float = 0.15) -> np.ndarray:
    """
    Applies pure NumPy/SciPy-based time warping to a 3D sequential array.
    Uses anchor-based interpolation to speed up or slow down segments of each sequence.
    Vectorized across features to eliminate inner loops.
    """
    N, L, D = X.shape
    X_warped = np.zeros_like(X)
    
    t = np.linspace(0, 1, L)
    orig_anchors = np.linspace(0, 1, num_anchors)
    
    for i in range(N):
        # Generate monotonic random walk for temporal warping anchors
        random_offsets = np.random.normal(0, sigma, num_anchors)
        warped_anchors = orig_anchors + random_offsets
        
        # Clamp endpoints to preserve sequence boundaries
        warped_anchors[0] = 0.0
        warped_anchors[-1] = 1.0
        
        # Ensure strict monotonicity
        for j in range(1, num_anchors - 1):
            warped_anchors[j] = max(warped_anchors[j-1] + 1e-5, min(warped_anchors[j], warped_anchors[j+1] - 1e-5))
            
        # Create temporal mapping function
        warp_fn = interp1d(orig_anchors, warped_anchors, kind='linear', fill_value="extrapolate")
        new_t = warp_fn(t)
        new_t = np.clip(new_t, 0.0, 1.0)
        
        # Interpolate all D channels simultaneously along axis=0
        interp_fn = interp1d(t, X[i], axis=0, kind='linear', fill_value="extrapolate")
        X_warped[i] = interp_fn(new_t)
            
    return X_warped


def augment_sequences(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies time warping to the sequences.
    
    Args:
        X (np.ndarray): Input sequence array of shape (N, L, D).
        y (np.ndarray): Labels of shape (N,).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Combined original and warped sequences.
    """
    if X.size == 0:
        return X, y
        
    if HAS_TSAUG:
        try:
            # We warp with tsaug
            augmenter = TimeWarp()
            X_warped = augmenter.augment(X)
        except Exception:
            # Fallback to pure numpy in case of runtime errors
            X_warped = _pure_numpy_time_warp(X)
    else:
        X_warped = _pure_numpy_time_warp(X)
        
    X_combined = np.vstack([X, X_warped])
    y_combined = np.hstack([y, y])
    
    return X_combined, y_combined


def augment_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Applies speed warping to flat telemetry sequences via spline interpolation.
    
    Args:
        df (pd.DataFrame): Flat telemetry DataFrame.
        target_col (str): Name of the target label column.
        
    Returns:
        pd.DataFrame: Augmented DataFrame (appended to original).
    """
    # Simply duplicate with custom index scaling as a flat warping fallback
    # For a flat DataFrame, we can simulate warping by selecting random operational rows
    n_rows = len(df)
    t = np.linspace(0, 1, n_rows)
    
    # Warping function
    orig_anchors = np.linspace(0, 1, 5)
    warped_anchors = orig_anchors + np.random.normal(0, 0.08, 5)
    warped_anchors[0], warped_anchors[-1] = 0.0, 1.0
    for j in range(1, 4):
        warped_anchors[j] = max(warped_anchors[j-1] + 1e-4, min(warped_anchors[j], warped_anchors[j+1] - 1e-4))
        
    warp_fn = interp1d(orig_anchors, warped_anchors, kind='linear')
    new_t = warp_fn(t)
    
    # Map back to indices
    warped_indices = np.clip(np.round(new_t * (n_rows - 1)).astype(int), 0, n_rows - 1)
    warped_df = df.iloc[warped_indices].copy()
    
    # Regenerate monotonic timestamp index
    time_cols = [col for col in df.columns if 'time' in col.lower() or col.lower() == 't']
    if time_cols:
        col = time_cols[0]
        step = df[col].iloc[1] - df[col].iloc[0] if len(df) > 1 else 0.008
        start_time = df[col].max() + step
        warped_df[col] = start_time + np.arange(n_rows) * step
        
    return pd.concat([df, warped_df], ignore_index=True)
