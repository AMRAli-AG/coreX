import sqlite3
from datetime import datetime
import random
import time

# =============== DATABASE SETUP ================
conn = sqlite3.connect("joints_test.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS joint_data (
    timestamp TEXT,

    temp_j1 REAL, temp_j2 REAL, temp_j3 REAL,
    temp_j4 REAL, temp_j5 REAL, temp_j6 REAL,

    joint_voltage_j1 REAL, joint_voltage_j2 REAL, joint_voltage_j3 REAL,
    joint_voltage_j4 REAL, joint_voltage_j5 REAL, joint_voltage_j6 REAL,

    robot_voltage REAL,

    effort_j1 REAL, effort_j2 REAL, effort_j3 REAL,
    effort_j4 REAL, effort_j5 REAL, effort_j6 REAL,

    payload REAL,

    tcp_fx REAL, tcp_fy REAL, tcp_fz REAL,
    tcp_tx REAL, tcp_ty REAL, tcp_tz REAL
)
""")
conn.commit()


# ============== INSERT DATA FUNCTION ================
def insert_sensor_data(data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    flat_data = [
        timestamp,

        *data["temps"],
        *data["joint_voltages"],
        data["robot_voltage"],
        *data["efforts"],
        data["payload"],
        *data["tcp"]
    ]

    if len(flat_data) != 27:
        raise ValueError(f"Expected 27 values, got {len(flat_data)}")

    cursor.execute("""
        INSERT INTO joint_data VALUES (
            ?,
            ?,?,?,?,?,?,
            ?,?,?,?,?,?,
            ?,
            ?,?,?,?,?,?,
            ?,
            ?,?,?,?,?,?
        )
    """, flat_data)

    conn.commit()

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


# ============== DATA COLLECTION LOOP ================
try:
    for _ in range(10):
        sensor_data = read_sensors_mock()
        insert_sensor_data(sensor_data)
        print("Data inserted:", sensor_data)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    conn.close()
    print("Test database closed.")

# To run this script: python test_robot_db.py