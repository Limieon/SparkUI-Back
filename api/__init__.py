
from config import SparkUIConfig as Config

from fastapi import FastAPI

app = FastAPI(title="SparkUI")

def init():
    import uvicorn
    uvicorn.run(app, host = Config.API.HOST, port = Config.API.PORT)