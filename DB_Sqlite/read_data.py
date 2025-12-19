# ---  Fetch and Print All Table Rows Individually ---

# import sqlite3
# conn = sqlite3.connect("joints_test.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM joint_data")
# rows = cursor.fetchall()
# for row in rows:
#     print(row)
# conn.close()



# --- Using pandas to read and display the data in Tabular Format ---

import sqlite3
import pandas as pd

conn = sqlite3.connect("joints_test.db")
df = pd.read_sql_query("SELECT * FROM joint_data", conn)
print(df.head(10))

conn.close()

# python read_data.py