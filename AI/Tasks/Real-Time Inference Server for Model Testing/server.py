import asyncio
import websockets
import json
from datetime import datetime

def simple_diagnosis(data):
    '''
    Simple diagnosis based on timestamp second parity
    '''
    return {
        "timestamp": data["timestamp"],
        "status": "Even" if datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S").second % 5 == 0 else "Odd"
    }

async def handler(websocket):
    '''
    Handle incoming client connections and messages
    '''
    print("Client connected")
    try:
        async for message in websocket:
            # message is JSON string
            data = json.loads(message) # parse JSON to dict
            result = simple_diagnosis(data)
            await websocket.send(json.dumps(result)) # send diagnosis back as JSON string
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Server running at ws://localhost:8765")
        await asyncio.Future()  # run forever

asyncio.run(main())
