# ==============================================
# Robot Joints Data Logging using InfluxDB
# ==============================================

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import random
import time

# =============== INFLUXDB SETUP ================
# Replace these values with your InfluxDB info
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "wVrs_5qA89sTPd3Rt1FFx4zpZykxa5L2KN4H9CrJLiIeGWBVS_iUVpHhR0_gXy6laJ4xwvvcSUwHqFFsffqUZQ=="
INFLUX_ORG = "test_robot_db"
INFLUX_BUCKET = "joint_data"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# =============== INSERT DATA FUNCTION ================
def insert_sensor_data(data):
    point = Point("joint_data").tag("robot", "robot1").time(datetime.utcnow())

    # Add fields
    for i, temp in enumerate(data["temps"], 1):
        point.field(f"temp_j{i}", temp)

    for i, volt in enumerate(data["joint_voltages"], 1):
        point.field(f"joint_voltage_j{i}", volt)

    point.field("robot_voltage", data["robot_voltage"])

    for i, effort in enumerate(data["efforts"], 1):
        point.field(f"effort_j{i}", effort)

    point.field("payload", data["payload"])

    tcp_labels = ["fx", "fy", "fz", "tx", "ty", "tz"]
    for label, val in zip(tcp_labels, data["tcp"]):
        point.field(f"tcp_{label}", val)

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
    print("Data inserted:", data)


# =============== MOCK DATA (TESTING) ================
def read_sensors_mock():
    return {
        "temps": [round(random.uniform(30, 80), 2) for _ in range(6)],
        "joint_voltages": [round(random.uniform(20, 48), 2) for _ in range(6)],
        "robot_voltage": round(random.uniform(20, 48), 2),
        "efforts": [round(random.uniform(0, 100), 2) for _ in range(6)],
        "payload": round(random.uniform(0, 10), 2),
        "tcp": [round(random.uniform(-50, 50), 2) for _ in range(6)]
    }

# =============== DATA COLLECTION LOOP ================
try:
    for _ in range(10):
        sensor_data = read_sensors_mock()
        insert_sensor_data(sensor_data)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    client.close()
    print("InfluxDB client closed.")

# To run this script: python test_robot_db.py