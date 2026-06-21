import ssl
import paho.mqtt.client as mqtt


HOST = "37e15fa95a3441108c3ac7989d2a08a4.s1.eu.hivemq.cloud"
PORT = 8883

USERNAME = "Subtest"
PASSWORD = "0000aA0000"

TOPIC = "robot/telemetry"

def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected")
    client.subscribe(TOPIC, qos=1)

def on_message(client, userdata, msg):
    value = msg.payload.decode()
    print("Received:", value)

client = mqtt.Client(protocol=mqtt.MQTTv5)

client.on_connect = on_connect
client.on_message = on_message

client.tls_set(tls_version=ssl.PROTOCOL_TLS)

client.username_pw_set(USERNAME,PASSWORD)

client.connect(HOST, PORT)

client.loop_forever()