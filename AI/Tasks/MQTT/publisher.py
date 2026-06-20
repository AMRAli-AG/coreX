import paho.mqtt.client as mqtt
import pandas as pd
import json
import time
import ast


BROKER = "localhost"
PORT = 1883
TOPIC = "robot/input"

CSV_PATH = "AI\\Datasets\\rtde_data.csv"

def connect_client():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER, PORT)
    client.loop_start()
    return client

def load(csv_path):
    df = pd.read_csv(csv_path)
    columns_to_expand = [
        "actual_current",
        "target_current",
        "joint_temperatures",
        "target_moment"
    ]
    
    for col in columns_to_expand:
        expanded = (df[col].apply(ast.literal_eval).apply(pd.Series))
        expanded.columns = [f"{col}_{i}"for i in range(6)]
        df = pd.concat([df, expanded], axis=1)
    for i in range(6):
        df[f"track_err_I_{i}"] = (df[f"actual_current_{i}"] - df[f"target_current_{i}"])
        
    model_input = []
    model_input += [f"actual_current_{i}" for i in range(6)]
    model_input += [f"joint_temperatures_{i}" for i in range(6)]
    model_input += [f"target_moment_{i}" for i in range(6)]
    model_input += [f"track_err_I_{i}" for i in range(6)]

    for _, row in df.iterrows():
        yield (row[model_input].to_dict())

def publish_data(client, payload):
    result = client.publish(TOPIC, json.dumps(payload))
    if result.rc == 0:
        print(f"Sent-> {payload}")

def main():
    client = connect_client()
    try:
        for payload in load(CSV_PATH):
            publish_data(client, payload)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nPublisher stopped")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()