from fastapi import FastAPI
import logging

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()
app = FastAPI(name="SparkUI - API")

logger: logging.Logger = None


@app.on_event("startup")
async def on_startup():
    global logger

    logger = logging.getLogger("uvicorn.info")
