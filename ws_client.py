import asyncio
import websockets
import json

SERVER_URI = "ws://localhost:8080/ws"  # Пример: "ws://localhost:8765"

async def hello_world(websocket):
    payload = {
        "type": "info",
        "message": "Hello World!!!!"
    }
    await websocket.send(json.dumps(payload))
    response = await websocket.recv()
    print(f"[INFO] Result response: {response}")

async def main():
    async with websockets.connect(SERVER_URI) as websocket:
        await hello_world(websocket)

if __name__ == "__main__":
    asyncio.run(main())
