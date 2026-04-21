import asyncio
import websockets
import json
import random
from datetime import datetime

SEND_DELAY = 1 # seconds

def generate_data():
    '''
    Generate simulated telemetry data
    ''' 
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": [40 + random.uniform(-1, 1) for _ in range(6)],
        "effort": [0.10 + random.uniform(-0.02, 0.02) for _ in range(6)],
        "voltage": [48 + random.uniform(-0.5, 0.5) for _ in range(6)]
    }

async def main():
    uri = "ws://localhost:8765" # server address
    print("Connecting...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server")
            while True:
                data = generate_data()
                await websocket.send(json.dumps(data)) # send JSON string
                response = await websocket.recv() # wait for server response
                print("time:", json.loads(response)['timestamp'])
                print("status:", json.loads(response)['status'])
                await asyncio.sleep(SEND_DELAY) # control send frequency
    except Exception as e:
        print("Connection error:", e)

asyncio.run(main())
