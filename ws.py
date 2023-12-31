import websockets


async def on(websocket, path):
    try:
        async for message in websocket:
            # Echo the received message back to the client
            await websocket.send(message)
    except websockets.exceptions.ConnectionClosedError:
        pass  # Connection was closed by the client
