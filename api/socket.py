import time
import json

from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


sockets = []


async def broadcast(data: str):
    for ws in sockets:
        await ws.send_json(data)


async def on_connect(ws: WebSocket):
    await ws.accept()
    sockets.append(ws)
    try:
        while True:
            await ws.receive()
    except:
        sockets.remove(ws)
