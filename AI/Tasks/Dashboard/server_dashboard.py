"""
Enhanced WebSocket Server with ML-based Anomaly Detection
Integrates with trained autoencoder model for real-time anomaly detection
"""

import asyncio
import websockets
import json
from datetime import datetime
import os
import sys
import warnings
import numpy as np
import requests

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
import joblib

# Flask API endpoint to update dashboard
FLASK_UPDATE_URL = "http://localhost:5000/api/update"

# Model paths (adjust based on your actual paths)
MODEL_PATH = "AI/Models/autoencoder/autoencoder_model.keras"
SCALER_PATH = "AI/Models/autoencoder/scaler.pkl"
THRESHOLD_PATH = "AI/Models/autoencoder/threshold.txt"

# Load trained artifacts
try:
    autoencoder = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(THRESHOLD_PATH) as f:
        BASE_THRESHOLD = float(f.read())
    
    model_input_dim = autoencoder.input_shape[1]
    
    print("\n[SERVER] ✓ Model loaded successfully")
    print(f"[SERVER] ✓ Input dimension: {model_input_dim}")
    print(f"[SERVER] ✓ Threshold: {BASE_THRESHOLD:.6f}")
except Exception as e:
    print(f"[SERVER] ✗ Error loading model: {e}")
    print("[SERVER] Model file paths:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {THRESHOLD_PATH}")
    print("\n[SERVER] Make sure:")
    print("  1. Model files exist in AI/Models/autoencoder/")
    print("  2. You exported the model from your Jupyter notebook")
    print("  3. Run the notebook cell 10 to export: autoencoder.save(...)")
    sys.exit(1)


def detect_anomaly(point_array):
    """
    Detect if a data point is an anomaly using the trained autoencoder.
    
    Args:
        point_array: numpy array or list of features (matches model input dimension)
        
    Returns:
        dict with error, is_anomaly, and other metrics
    """
    try:
        # Reshape to (1, num_features)
        point = np.asarray(point_array, dtype=np.float32).reshape(1, -1)
        
        # Validate input shape matches model
        expected_features = autoencoder.input_shape[1]
        if point.shape[1] != expected_features:
            error_msg = f"Feature mismatch! Model expects {expected_features} features, got {point.shape[1]}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] Received features: {point_array}")
            raise ValueError(error_msg)
        
        print(f"[DEBUG] Features validated: {expected_features}")
        
        # Normalize using the scaler
        point_norm = scaler.transform(point)
        print(f"[DEBUG] Data normalized")
        
        # Get reconstruction
        reconstructed = autoencoder.predict(point_norm, verbose=0)
        print(f"[DEBUG] Reconstruction complete")
        
        # Calculate reconstruction error (MSE)
        error = float(np.mean((point_norm - reconstructed) ** 2))
        
        # Determine if anomaly
        is_anomaly = error > BASE_THRESHOLD
        
        # Calculate confidence score
        if is_anomaly:
            # How much above threshold
            distance_from_threshold = error - BASE_THRESHOLD
            confidence = min(100, (distance_from_threshold / BASE_THRESHOLD * 100))
        else:
            # How much below threshold
            distance_from_threshold = BASE_THRESHOLD - error
            confidence = min(100, (distance_from_threshold / BASE_THRESHOLD * 100))
        
        return {
            "error": error,
            "threshold": BASE_THRESHOLD,
            "is_anomaly": is_anomaly,
            "status": "ANOMALY" if is_anomaly else "NORMAL",
            "confidence": confidence
        }
    except Exception as e:
        print(f"[ERROR] Anomaly detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": 0,
            "threshold": BASE_THRESHOLD,
            "is_anomaly": False,
            "status": "ERROR",
            "message": str(e),
            "confidence": 0
        }


async def handler(websocket):
    """
    Handle incoming WebSocket connections and perform real-time anomaly detection.
    """
    client_addr = websocket.remote_address
    print(f"[SERVER] Client connected from {client_addr}")
    print(f"[SERVER] Model input dimension: {autoencoder.input_shape[1]}")
    
    try:
        async for message in websocket:
            try:
                # Parse incoming JSON
                data = json.loads(message)
                
                # Extract features - support both old and new format
                if "features" in data:
                    # New format: direct feature list
                    point_features = data.get("features", [])
                elif "temperature" in data and "effort" in data and "voltage" in data:
                    # Old format: temperature + effort + voltage arrays
                    temperatures = data.get("temperature", [])
                    efforts = data.get("effort", [])
                    voltages = data.get("voltage", [])
                    point_features = temperatures + efforts + voltages
                else:
                    raise ValueError("No valid features found in data")
                
                print(f"[DEBUG] Received {len(point_features)} features")
                
                # Perform anomaly detection
                detection_result = detect_anomaly(point_features)
                
                # Prepare response
                response = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "status": detection_result["status"],
                    "error": round(detection_result["error"], 6),
                    "threshold": round(detection_result["threshold"], 6),
                    "is_anomaly": detection_result["is_anomaly"],
                    "confidence": round(detection_result["confidence"], 2),
                    "received_data": {
                        "features": point_features
                    }
                }
                
                # Send response back to client
                await websocket.send(json.dumps(response))
                
                # Also send to Flask to update dashboard
                try:
                    requests.post(FLASK_UPDATE_URL, json=response, timeout=2)
                except Exception as e:
                    print(f"[ERROR] Failed to update Flask: {e}")
                
                # Log to console
                status_icon = "🔴" if detection_result["is_anomaly"] else "🟢"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status_icon} {detection_result['status']} | Error: {detection_result['error']:.6f} | Confidence: {detection_result['confidence']:.1f}%")
                
            except json.JSONDecodeError:
                print(f"[ERROR] Invalid JSON received: {message[:50]}")
                await websocket.send(json.dumps({
                    "status": "ERROR",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                print(f"[ERROR] Processing error: {e}")
                await websocket.send(json.dumps({
                    "status": "ERROR",
                    "message": str(e)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"[SERVER] Client {client_addr} disconnected")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


async def main():
    """Start the WebSocket server."""
    try:
        async with websockets.serve(handler, "localhost", 8765):
            print("\n" + "="*60)
            print("🚀 Real-Time Anomaly Detection Server")
            print("="*60)
            print("📡 WebSocket Server running at: ws://localhost:8765")
            print("📊 Dashboard URL: http://localhost:5000")
            print("="*60 + "\n")
            await asyncio.Future()  # run forever
    except OSError as e:
        print(f"[ERROR] Could not start server: {e}")
        print("Make sure port 8765 is available")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down gracefully...")
        sys.exit(0)
