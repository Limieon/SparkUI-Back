import os
import asyncio

import ws

import websockets.server
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import SparkUIConfig as Config

from db import DB

app = FastAPI(title="SparkUI")
db = DB()


@app.websocket("/")
async def socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        pass


async def run_app():
    import uvicorn

    uvicorn_config = uvicorn.Config(app, host=Config.API.HOST, port=Config.API.PORT)
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


async def init():
    print("Initializing directories...")
    for field, value in vars(Config.StableDiffusion.Directories).items():
        if field.startswith("__"):
            continue
        path = os.path.join(".", value)
        if not os.path.exists(path):
            os.makedirs(path)

    import api.v1.stable_diffusion

    print("Initializing database...")
    await db.connect()

    print("Initializing middlewares...")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"{Config.Frontend.HOST}:{Config.Frontend.PORT}"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    print("Initializing routes...")
    app.include_router(
        api.v1.stable_diffusion.init(),
        prefix="/v1/stable_diffusion",
        tags=["StableDiffusion"],
    )


async def init_socket():
    print("Initializing socket...")
    websockets.server.serve(ws.on, Config.API.HOST, Config.API.PORT)


async def shutdown():
    print("Shutting down database...")
    await db.shutdown()

    print("Shutdown successful!")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(init())
        # loop.run_until_complete(init_socket())
        loop.run_until_complete(run_app())
    finally:
        loop.run_until_complete(shutdown())
        loop.close()
