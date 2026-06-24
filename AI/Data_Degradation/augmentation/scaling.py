# -*- coding: utf-8 -*-
"""
Magnitude Scaling Data Augmentation Technique.
Grounded in operational physics where signal amplification or calibration shifts occur.
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Standard technique identification
TECHNIQUE_NAME = "Magnitude Scaling"


def augment_sequences(
    X: np.ndarray, y: np.ndarray, scaling_range: Tuple[float, float] = (0.9, 1.1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies random scaling factors to individual feature sequences.
    
    Args:
        X (np.ndarray): Input sequence array of shape (N, L, D).
        y (np.ndarray): Labels of shape (N,).
        scaling_range (Tuple[float, float]): Min and max scaling factor range.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Combined original and scaled sequences.
    """
    if X.size == 0:
        return X, y
        
    N, L, D = X.shape
    
    # Generate random scaling factor for each sample and feature
    factors = np.random.uniform(scaling_range[0], scaling_range[1], size=(N, 1, D))
    X_scaled = X * factors
    
    X_combined = np.vstack([X, X_scaled])
    y_combined = np.hstack([y, y])
    
    return X_combined, y_combined


def augment_dataframe(
    df: pd.DataFrame, target_col: str, scaling_range: Tuple[float, float] = (0.95, 1.05)
) -> pd.DataFrame:
    """
    Applies scaling factors feature-wise to the entire DataFrame.
    Vectorized implementation with zero Python loops.
    
    Args:
        df (pd.DataFrame): Flat telemetry DataFrame.
        target_col (str): Name of the target label column.
        scaling_range (Tuple[float, float]): Range of scale factors.
        
    Returns:
        pd.DataFrame: Augmented DataFrame (appended to original).
    """
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Exclude labels and time indices from scaling
    cols_to_scale = [
        col for col in df_numeric.columns 
        if col != target_col and 'time' not in col.lower() and col.lower() != 't'
    ]
    
    scaled_df = df.copy()
    if cols_to_scale:
        # Vectorized scaling factors for all columns simultaneously
        factors = np.random.uniform(scaling_range[0], scaling_range[1], size=len(cols_to_scale))
        scaled_df[cols_to_scale] = scaled_df[cols_to_scale] * factors
        
    return pd.concat([df, scaled_df], ignore_index=True)
