import paho.mqtt.client as mqtt
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import joblib

BROKER = "localhost"
PORT = 1883

INPUT_TOPIC = "robot/input"
OUTPUT_TOPIC = "robot/output"

# Load model
model = tf.keras.models.load_model("AI\Models\TCN\\best_model.keras")
scaler = joblib.load("AI\Models\TCN\scaler.pkl")
threshold = float(joblib.load("AI\Models\TCN\\threshold.pkl"))
def preprocess(payload):
    x = pd.DataFrame([payload])
    x = scaler.transform(x)
    return x

def run_inference(payload):
    x = preprocess(payload)
    pred = model.predict(x,verbose=0)
    mse = np.mean(np.square(x - pred))
    is_anomaly = (mse > threshold)
    result = {
        "anomaly_score":float(mse),
        "threshold":float(threshold),
        "status":("Critical" if is_anomaly else "Healthy")
    }
    return result

def on_message(client, userdata,msg):
    payload = json.loads(msg.payload.decode())
    print("\nReceived")
    result = run_inference(payload)
    print(result)
    client.publish(OUTPUT_TOPIC,json.dumps(result))

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = (on_message)
    client.connect(BROKER, PORT)
    client.subscribe(INPUT_TOPIC)
    print(f"Listening on {INPUT_TOPIC}")
    client.loop_forever()

if __name__ == "__main__":
    main()