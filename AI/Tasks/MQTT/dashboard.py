import paho.mqtt.client as mqtt
import json
import os

BROKER = "localhost"
PORT = 1883
TOPIC = "robot/output"

def clear():os.system("cls" if os.name == "nt" else "clear")

def draw(score):
    width = 40
    filled = int(min(score, 1) * width)
    return ("█" * filled + "-" * (width - filled))

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    score = payload["anomaly_score"]
    status = payload["status"]
    threshold = payload["threshold"]
    clear()
    print("\n🤖 ROBOT HEALTH\n")
    print(f"Status     : {status}")
    print(f"Score      : {score:.5f}")
    print(f"Threshold  : {threshold:.5f}")
    print()
    ratio = score / threshold
    print(draw(ratio))
    print()
    if status == "Healthy":
        print("🟢 Normal")
    else:
        print("🔴 ANOMALY")

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = (on_message)
    client.connect(BROKER, PORT)
    client.subscribe(TOPIC)
    print("Listening...")
    client.loop_forever()

if __name__ == "__main__":
    main()