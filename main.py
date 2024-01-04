import os
import asyncio
import threading

import api.socket

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import SparkUIConfig as Config

from db import DB

from civitai.importer import importer_queue_step

app = FastAPI(title="SparkUI")
db = DB()

queue_stepper_stop = False


@app.websocket("/")
async def socket(ws: WebSocket):
    await api.socket.on_connect(ws)


async def run_webserver():
    import uvicorn

    uvicorn_config = uvicorn.Config(app, host=Config.API.HOST, port=Config.API.PORT)
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


def run_queue_stepper():
    async def run():
        while not queue_stepper_stop:
            await asyncio.sleep(1)
            await importer_queue_step()

    asyncio.run(run())


async def init():
    print("Initializing directories...")
    for field, value in vars(Config.StableDiffusion.Directories).items():
        if field.startswith("__"):
            continue
        path = os.path.join(".", value)
        if not os.path.exists(path):
            os.makedirs(path)

    import api.v1.stable_diffusion
    import api.v1.image

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
        api.v1.image.init(),
        prefix="/v1/image",
        tags=["Image"],
    )
    app.include_router(
        api.v1.stable_diffusion.init(),
        prefix="/v1/stable_diffusion",
        tags=["StableDiffusion"],
    )


async def shutdown():
    print("Shutting down database...")
    await db.shutdown()

    print("Shutdown successful!")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=run_queue_stepper)
    thread.start()

    loop.run_until_complete(init())
    loop.run_until_complete(run_webserver())
    loop.run_until_complete(shutdown())
    queue_stepper_stop = True
    thread.join()
