import os
import warnings

# Suppress TensorFlow and scikit-learn warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Load trained artifacts
autoencoder = tf.keras.models.load_model(
    "AI\Models\\autoencoder\\autoencoder_model.keras"
)
scaler = joblib.load("AI\Models\\autoencoder\\scaler.pkl")

with open("AI\Models\\autoencoder\\threshold.txt") as f:
    BASE_THRESHOLD = float(f.read())


def check_realtime_point(point):
    """
    Check a single real-time point for anomaly.
    """
    # Convert & reshape
    point = np.asarray(point).reshape(1, -1)

    # Normalize
    point_norm = scaler.transform(point)

    # Reconstruct
    reconstructed = autoencoder.predict(point_norm, verbose=0)

    # Reconstruction error
    error = np.mean((point_norm - reconstructed) ** 2)

    # Conservative threshold
    threshold = BASE_THRESHOLD

    # Decision
    is_anomaly = error > threshold

    return error, is_anomaly


def detect_anomalies_in_dataset(df, time_column='Time'):
    # Load and remove time column
    df = df.copy()
    if time_column in df.columns:
        df = df.drop(columns=[time_column])
    
    anomaly_indices = []
    
    # Check each point in the dataset
    for idx, row in df.iterrows():
        point = row.values
        error, is_anomaly = check_realtime_point(point)
        
        if is_anomaly:
            anomaly_indices.append(idx)
            print(f"Anomaly detected at index: {idx}")
    
    print(f"\nTotal anomalies found: {len(anomaly_indices)}")
    return anomaly_indices


def add_noise_to_dataset(csv_path, num_noisy_rows, noise_level=0.1, time_column='Time'):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Make a copy to modify
    df_noisy = df.copy()
    
    # Get total number of rows
    total_rows = len(df)
    
    # Validate input
    if num_noisy_rows > total_rows:
        print(f"Warning: num_noisy_rows ({num_noisy_rows}) exceeds total rows ({total_rows})")
        num_noisy_rows = total_rows
    
    # Randomly select rows to add noise to
    noisy_indices = np.random.choice(total_rows, size=num_noisy_rows, replace=False)
    noisy_indices = sorted(noisy_indices)
    
    # Add noise to selected rows and all numeric columns
    for idx in noisy_indices:
        for col in df_noisy.columns:
            # Skip time column
            if col == time_column:
                continue
            
            # Add Gaussian noise to numeric columns
            if pd.api.types.is_numeric_dtype(df_noisy[col]):
                original_value = df_noisy.at[idx, col]
                noise = np.random.normal(0, noise_level * abs(original_value) if original_value != 0 else noise_level)
                df_noisy.at[idx, col] = original_value + noise
    
    print(f"Total rows with noise added: {len(noisy_indices)}")
    print(f"Row indices with noise: {noisy_indices}")
    
    return df_noisy, noisy_indices


if __name__ == "__main__":
    # Example usage
    test_csv_path = "AI/Datasets/matlab_v1.csv"
    new_data, noisy_rows = add_noise_to_dataset(test_csv_path, num_noisy_rows=10, noise_level=0.5)
    anomalies = detect_anomalies_in_dataset(new_data)
    print("Anomaly indices:", anomalies)
