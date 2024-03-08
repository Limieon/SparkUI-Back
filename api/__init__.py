from fastapi import FastAPI

import api.v1.stable_diffusion


async def serve(app: FastAPI, host: str, port: int):
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)

    api.v1.stable_diffusion.init_routes(app)

    await server.serve()
