"""
Flask Web Application for Real-Time Anomaly Detection Dashboard
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from datetime import datetime
from collections import deque
import threading

app = Flask(__name__)
CORS(app)

# Store data for visualization
data_store = {
    "latest": None,
    "history": deque(maxlen=100),  # Keep last 100 records
    "anomaly_count": 0,
    "normal_count": 0,
    "error_history": deque(maxlen=100),
}

data_lock = threading.Lock()


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/update', methods=['POST'])
def update_detection():
    """
    Receive detection results from WebSocket server and store them.
    Called by server_dashboard.py to update the dashboard.
    """
    try:
        detection_result = request.get_json()
        
        with data_lock:
            data_store["latest"] = detection_result
            data_store["history"].append({
                "timestamp": detection_result.get("timestamp"),
                "status": detection_result.get("status"),
                "error": detection_result.get("error"),
                "is_anomaly": detection_result.get("is_anomaly")
            })
            data_store["error_history"].append(detection_result.get("error", 0))
            
            if detection_result.get("is_anomaly"):
                data_store["anomaly_count"] += 1
            else:
                data_store["normal_count"] += 1
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"[FLASK] Error updating detection: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/api/latest')
def get_latest():
    """Get the latest detection result."""
    with data_lock:
        return jsonify(data_store["latest"] or {})


@app.route('/api/history')
def get_history():
    """Get historical data for charting."""
    with data_lock:
        return jsonify({
            "history": list(data_store["history"]),
            "error_history": list(data_store["error_history"]),
            "anomaly_count": data_store["anomaly_count"],
            "normal_count": data_store["normal_count"],
        })


@app.route('/api/stats')
def get_stats():
    """Get statistical information."""
    with data_lock:
        total = data_store["anomaly_count"] + data_store["normal_count"]
        anomaly_rate = (data_store["anomaly_count"] / total * 100) if total > 0 else 0
        
        return jsonify({
            "total_samples": total,
            "anomalies": data_store["anomaly_count"],
            "normal": data_store["normal_count"],
            "anomaly_rate": round(anomaly_rate, 2),
            "uptime": datetime.now().isoformat()
        })


def update_data(detection_result):
    """
    Called by the WebSocket server to update dashboard data.
    Call this function after receiving a detection result.
    """
    with data_lock:
        data_store["latest"] = detection_result
        data_store["history"].append({
            "timestamp": detection_result.get("timestamp"),
            "status": detection_result.get("status"),
            "error": detection_result.get("error"),
            "is_anomaly": detection_result.get("is_anomaly")
        })
        data_store["error_history"].append(detection_result.get("error", 0))
        
        if detection_result.get("is_anomaly"):
            data_store["anomaly_count"] += 1
        else:
            data_store["normal_count"] += 1


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Flask Dashboard Server")
    print("="*60)
    print("📊 Dashboard URL: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host='localhost', port=5000)
