"""
Real-Time Data Client for Anomaly Detection Dashboard
Sends actual robot joint data to the WebSocket server for real-time anomaly detection
Loads data from the matlab_v1.csv dataset and sends it to the model for inference
"""

import asyncio
import websockets
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Configuration
SEND_DELAY = 1  # seconds between data transmissions
SERVER_HOST = "localhost"
SERVER_PORT = 8765
WEBSOCKET_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"

# Dataset path
DATASET_PATH = "AI/Datasets/matlab_v1.csv"
GITHUB_URL = "https://raw.githubusercontent.com/AMRAli-AG/coreX/refs/heads/ai/AI/Datasets/matlab_v1.csv"


def load_dataset(local_path=DATASET_PATH, github_url=GITHUB_URL):
    """
    Load the matlab_v1.csv dataset from local file or GitHub.
    Preprocesses the data to match model training format.
    """
    df = None
    
    # Try local file first
    if Path(local_path).exists():
        print(f"[DATA] Loading from local file: {local_path}")
        try:
            df = pd.read_csv(local_path)
            print(f"[DATA] ✓ Loaded {len(df)} rows from local file")
        except Exception as e:
            print(f"[DATA] ✗ Error loading local file: {e}")
            df = None
    
    # Fall back to GitHub if local not available
    if df is None:
        print(f"[DATA] Loading from GitHub...")
        try:
            df = pd.read_csv(github_url)
            print(f"[DATA] ✓ Loaded {len(df)} rows from GitHub")
        except Exception as e:
            print(f"[DATA] ✗ Error loading from GitHub: {e}")
            print("[DATA] ✗ Make sure you have internet connection or local CSV file")
            return None
    
    # Preprocess data
    df_proc = df.copy()
    
    # Drop time column if exists
    time_cols = [c for c in df_proc.columns if 'time' in c.lower()]
    if time_cols:
        print(f"[DATA] Dropping time columns: {time_cols}")
        df_proc = df_proc.drop(columns=time_cols)
    
    # Ensure all columns are numeric
    for col in df_proc.columns:
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
    
    # Drop rows with NaN
    initial_rows = len(df_proc)
    df_proc = df_proc.dropna()
    if len(df_proc) < initial_rows:
        print(f"[DATA] Dropped {initial_rows - len(df_proc)} rows with NaN values")
    
    print(f"[DATA] Final shape: {df_proc.shape}")
    print(f"[DATA] Features: {df_proc.columns.tolist()}")
    
    return df_proc


async def main():
    """Connect to server and send real data."""
    uri = WEBSOCKET_URL
    
    print("\n" + "="*70)
    print("📊 Real-Time Data Client - Robot Joint Anomaly Detection")
    print("="*70)
    print(f"Connecting to: {uri}")
    print(f"Send delay: {SEND_DELAY} second(s)")
    print("="*70 + "\n")

    # Load dataset
    print("[INIT] Loading dataset...")
    df = load_dataset()
    
    if df is None:
        print("\n[ERROR] Failed to load dataset. Exiting.")
        sys.exit(1)
    
    num_features = df.shape[1]
    print(f"[INIT] Dataset ready with {num_features} features\n")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to server\n")
            
            message_count = 0
            anomaly_count = 0
            current_idx = 0
            
            while True:
                # Get next data point from dataset
                if current_idx >= len(df):
                    current_idx = 0  # Restart from beginning
                    print("\n[INFO] Restarting data stream from beginning...\n")
                
                row = df.iloc[current_idx].values.tolist()
                current_idx += 1
                
                # Occasionally inject noise to simulate anomalies (10% chance)
                if random.random() < 0.1:
                    row_to_send = [x + random.gauss(0, abs(x) * 0.2) if x != 0 else random.gauss(0, 0.1) 
                                   for x in row]
                    anomaly_count += 1
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 Sending ANOMALOUS data (#{message_count + 1})")
                else:
                    row_to_send = row
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 Sending normal data (#{message_count + 1})")
                
                data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "features": row_to_send
                }
                
                # Send data to server
                await websocket.send(json.dumps(data))
                message_count += 1
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    # Display response
                    status_icon = "🔴" if response_data.get('is_anomaly') else "🟢"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status_icon} Response: {response_data['status']}")
                    print(f"    Error: {response_data['error']:.6f} | Threshold: {response_data['threshold']:.6f}")
                    print(f"    Confidence: {response_data['confidence']:.1f}%\n")
                    
                except asyncio.TimeoutError:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  No response from server (timeout)\n")
                
                # Wait before next transmission
                await asyncio.sleep(SEND_DELAY)
                
    except ConnectionRefusedError:
        print(f"❌ ERROR: Could not connect to {uri}")
        print("Make sure the server is running:")
        print("  python server_dashboard.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n[✓] Client stopped by user")
        print(f"Statistics:")
        print(f"  - Total messages sent: {message_count}")
        print(f"  - Anomalous messages injected: {anomaly_count}")
        if message_count > 0:
            print(f"  - Anomaly injection rate: {anomaly_count/message_count*100:.1f}%")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Gracefully shut down")
        sys.exit(0)
