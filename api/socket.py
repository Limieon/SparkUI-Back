import time
import json

import asyncio


from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


sockets = []


class SocketMessageID:
    civitai_importer_update = "civitai_importer_update"


async def sockets_broadcast(id: str, data: str):
    for ws in sockets:
        await ws.send_json({"id": id, "data": data})


async def on_connect(ws: WebSocket):
    await ws.accept()
    sockets.append(ws)

    try:
        while True:
            await ws.receive()
    except:
        sockets.remove(ws)
