from influxdb_client import InfluxDBClient
import pandas as pd

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "wVrs_5qA89sTPd3Rt1FFx4zpZykxa5L2KN4H9CrJLiIeGWBVS_iUVpHhR0_gXy6laJ4xwvvcSUwHqFFsffqUZQ=="
INFLUX_ORG = "test_robot_db"
INFLUX_BUCKET = "joint_data"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

query = f'''
from(bucket:"{INFLUX_BUCKET}")
|> range(start: -1h)
'''

tables = query_api.query(query)
data = []

for table in tables:
    for record in table.records:
        data.append(record.values)

df = pd.DataFrame(data)
print(df.head(20))
print(df.shape)
print("# # # #"*20, '\n')

# Pivot the DataFrame for better readability
df_wide = df.pivot(index="_time", columns="_field", values="_value")
print(df_wide.head(10))

# df_wide.to_csv("df_wide.csv", index=True)
# print("Save Done✅")

client.close()

# To run this script: python read_data.py