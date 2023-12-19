
from config import SparkUIConfig as Config

from fastapi import FastAPI

app = FastAPI(title="SparkUI")

def init():
    import api.v1.stable_diffusion
    
    app.include_router(api.v1.stable_diffusion.init(), prefix="/v1/stable_diffusion", tags=["StableDiffusion"])
    
    import uvicorn
    uvicorn.run(app, host = Config.API.HOST, port = Config.API.PORT)
