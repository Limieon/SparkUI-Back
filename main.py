import os
import asyncio

from fastapi import FastAPI

from config import SparkUIConfig as Config

from db import DB
app = FastAPI(title="SparkUI")
db = DB()

async def run_app():
    import uvicorn
    uvicorn_config = uvicorn.Config(app, host=Config.API.HOST, port=Config.API.PORT)
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

async def init():
    print("Initializing Directories...")
    for field, value in vars(Config.Directories.StableDiffusion).items():
        if field.startswith("__"): continue
        path = os.path.join(".", value)
        if not os.path.exists(path):
            os.makedirs(path)

    import api.v1.stable_diffusion

    print("Initializing database...")
    await db.connect()

    print("Initializing routes...")
    app.include_router(api.v1.stable_diffusion.init(), prefix="/v1/stable_diffusion", tags=["StableDiffusion"])

async def shutdown():
    print("Shutting down database...")
    await db.shutdown()

    print("Shutdown successful!")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(init())
        loop.run_until_complete(run_app())
    finally:
        loop.run_until_complete(shutdown())
        loop.close()
