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
import socket
import threading
import json
import time
from collections import deque

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


# ============================================================================
# REAL-TIME DATA STREAMING OVER ETHERNET (TCP SOCKET)
# ============================================================================
# Why TCP instead of UDP:
# - TCP ensures reliable data delivery (guaranteed packet order & arrival)
# - UDP is faster but can lose packets (unsuitable for sensor data anomaly detection)
# - TCP has built-in error checking and flow control
# ============================================================================

class RealtimeDataReceiver:
    """
    Non-blocking TCP Server to receive real-time sensor data over Ethernet.
    
    Data Format (Expected JSON):
        {"sensor_1": value1, "sensor_2": value2, ..., "sensor_n": valueN}
    """
    
    def __init__(self, host='0.0.0.0', port=5000, buffer_size=100):
        """
        Initialize the Real-time Data Receiver.
        
        Args:
            host (str): IP address to bind to (default: 0.0.0.0 = all interfaces)
            port (int): Port number to listen on (default: 5000)
            buffer_size (int): Max number of data points to keep in memory
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        
        # Thread-safe data buffer
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Server state
        self.is_running = False
        self.server_socket = None
        self.server_thread = None
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
    
    def start_server(self):
        """
        Start the TCP server in a background thread (non-blocking).
        """
        if self.is_running:
            print("⚠️  Server already running!")
            return
        
        self.is_running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"✅ TCP Server started on {self.host}:{self.port}")
    
    def _run_server(self):
        """
        Internal method: Run TCP server and accept connections.
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Non-blocking timeout
            
            print(f"🔌 Server listening on {self.host}:{self.port}")
            
            while self.is_running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"📡 Client connected: {client_address}")
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"❌ Error accepting client: {e}")
        
        except Exception as e:
            print(f"❌ Server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _handle_client(self, client_socket, client_address):
        """
        Internal method: Handle incoming data from a connected client.
        """
        try:
            while self.is_running:
                # Receive data (non-blocking with timeout)
                data = client_socket.recv(1024).decode('utf-8')
                
                if not data:
                    break
                
                # Parse JSON data
                try:
                    sensor_data = json.loads(data.strip())
                    
                    # Thread-safe: Add to buffer
                    with self.lock:
                        self.data_buffer.append(sensor_data)
                    
                    print(f"📥 Data received from {client_address}: {sensor_data}")
                
                except json.JSONDecodeError as e:
                    print(f"⚠️  Invalid JSON from {client_address}: {e}")
        
        except Exception as e:
            print(f"❌ Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
            print(f"🔌 Client disconnected: {client_address}")
    
    def stop_server(self):
        """
        Stop the TCP server gracefully.
        """
        if not self.is_running:
            print("⚠️  Server not running!")
            return
        
        self.is_running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        if self.server_thread:
            self.server_thread.join(timeout=2)
        
        print("✅ TCP Server stopped")
    
    def get_latest_data(self, num_points=1):
        """
        Retrieve latest data points from buffer.
        
        Args:
            num_points (int): Number of latest points to retrieve
        
        Returns:
            list: List of most recent data points
        """
        with self.lock:
            return list(self.data_buffer)[-num_points:]
    
    def process_realtime_data(self, callback=None):
        """
        Continuously process incoming data and detect anomalies.
        
        Args:
            callback (function): Optional function to call when anomaly detected.
                                 Signature: callback(anomaly_info)
        """
        print("🔄 Starting real-time anomaly detection...")
        
        while self.is_running:
            with self.lock:
                if len(self.data_buffer) == 0:
                    time.sleep(0.1)
                    continue
                
                # Get the latest point
                latest_data = self.data_buffer[-1]
            
            try:
                # Convert dict values to array for model
                values = np.array(list(latest_data.values())).reshape(1, -1)
                
                # Check for anomaly
                error, is_anomaly = check_realtime_point(values)
                
                if is_anomaly:
                    print(f"🚨 ANOMALY DETECTED! Error: {error:.4f}")
                    
                    if callback:
                        callback({
                            'timestamp': time.time(),
                            'data': latest_data,
                            'error': error,
                            'threshold': BASE_THRESHOLD
                        })
            
            except Exception as e:
                print(f"❌ Error processing data: {e}")
            
            time.sleep(0.1)  # Prevent busy-waiting


if __name__ == "__main__":
    # Example usage
    test_csv_path = "AI/Datasets/matlab_v1.csv"
    new_data, noisy_rows = add_noise_to_dataset(test_csv_path, num_noisy_rows=10, noise_level=0.5)
    anomalies = detect_anomalies_in_dataset(new_data)
    print("Anomaly indices:", anomalies)
    
    # ========================================================================
    # EXAMPLE: Real-Time Data Streaming
    # ========================================================================
    """
    1. Start the server:
        receiver = RealtimeDataReceiver(host='0.0.0.0', port=5000)
        receiver.start_server()
    
    2. Send test data from client (Python):
        import socket
        import json
        
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 5000))
        
        # Send sensor data as JSON
        data = {"sensor_1": 0.5, "sensor_2": 0.3, "sensor_3": 0.8}
        client.send(json.dumps(data).encode('utf-8'))
        client.close()
    
    3. Send test data from command line (PowerShell/Bash):
        echo '{"sensor_1": 0.5, "sensor_2": 0.3, "sensor_3": 0.8}' | nc localhost 5000
    
    4. Stop the server:
        receiver.stop_server()
    """
    
    # ========================================================================
    # UNCOMMENT TO TEST REAL-TIME DATA STREAMING:
    # ========================================================================
    
    # # Initialize the real-time data receiver
    # receiver = RealtimeDataReceiver(host='0.0.0.0', port=5000, buffer_size=100)
    
    # # Define a callback function for anomalies
    # def on_anomaly_detected(anomaly_info):
    #     print(f"\n🚨 ALERT: Anomaly detected!")
    #     print(f"   Error: {anomaly_info['error']:.4f}")
    #     print(f"   Threshold: {anomaly_info['threshold']:.4f}")
    #     print(f"   Data: {anomaly_info['data']}\n")
    
    # # Start the server
    # receiver.start_server()
    
    # # Start processing real-time data in background thread
    # processing_thread = threading.Thread(
    #     target=receiver.process_realtime_data,
    #     args=(on_anomaly_detected,),
    #     daemon=True
    # )
    # processing_thread.start()
    
    # # Keep the program running
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\n⏹️  Shutting down...")
    #     receiver.stop_server()
    #     print("✅ Done!")

