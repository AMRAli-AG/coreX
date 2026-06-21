import ssl
import json
import time
import random
import paho.mqtt.client as mqtt


HOST = "37e15fa95a3441108c3ac7989d2a08a4.s1.eu.hivemq.cloud"
PORT = 8883

USERNAME = ""
PASSWORD = ""

TOPIC = "robot/telemetry"

client = mqtt.Client(protocol=mqtt.MQTTv5)

client.tls_set(tls_version=ssl.PROTOCOL_TLS)

client.username_pw_set(USERNAME,PASSWORD)
client.connect(HOST, PORT)
client.loop_start()
print("Publishing Fake Data...")

while True:
    number = random.randint(0, 100)
    client.publish(TOPIC, str(str(number))) 
    print("Sent:", number)
    time.sleep(10)