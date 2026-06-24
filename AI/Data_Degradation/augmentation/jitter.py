# -*- coding: utf-8 -*-
"""
Jittering Data Augmentation Technique.
Grounded in statistical time-series augmentation by adding zero-mean Gaussian noise.
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Standard technique identification
TECHNIQUE_NAME = "Gaussian Jittering"


def augment_sequences(
    X: np.ndarray, y: np.ndarray, sigma: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies zero-mean Gaussian noise to sequential data.
    
    Args:
        X (np.ndarray): Input sequence array of shape (N, L, D).
        y (np.ndarray): Labels of shape (N,).
        sigma (float): Standard deviation of the Gaussian noise.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented X (original + jittered) and repeated labels y.
    """
    if X.size == 0:
        return X, y
        
    # Generate Gaussian noise matching X shape
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    X_jittered = X + noise
    
    # Combine original and augmented data
    X_combined = np.vstack([X, X_jittered])
    y_combined = np.hstack([y, y])
    
    return X_combined, y_combined


def augment_dataframe(df: pd.DataFrame, target_col: str, sigma: float = 0.03) -> pd.DataFrame:
    """
    Applies Gaussian noise directly to a pandas DataFrame for flat telemetry expansion.
    Vectorized implementation with zero Python loops.
    
    Args:
        df (pd.DataFrame): Flat telemetry DataFrame.
        target_col (str): Name of the target label column.
        sigma (float): Standard deviation of the noise relative to feature variance.
        
    Returns:
        pd.DataFrame: Augmented DataFrame (appended to original).
    """
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Exclude target and timestamp columns from noise injection
    cols_to_jitter = [
        col for col in df_numeric.columns 
        if col != target_col and 'time' not in col.lower() and col.lower() != 't'
    ]
    
    jittered_df = df.copy()
    if cols_to_jitter:
        # Calculate standard deviations vectorially
        stds = df[cols_to_jitter].std().values
        # Replace NaN or 0 standard deviations with 1.0 to avoid scaling collapse
        stds = np.where(np.isnan(stds) | (stds == 0.0), 1.0, stds)
        
        # Generate noise matching shape (len(df), len(cols_to_jitter))
        noise = np.random.normal(loc=0.0, scale=sigma * stds, size=(len(df), len(cols_to_jitter)))
        jittered_df[cols_to_jitter] = jittered_df[cols_to_jitter] + noise
        
    return pd.concat([df, jittered_df], ignore_index=True)
