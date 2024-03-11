from fastapi import FastAPI
import logging

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.pipeline_manager import PipelineManager, GenerationQueue

from prisma import Prisma

pipeline_manager = PipelineManager()
generation_queue = GenerationQueue(pipeline_manager)
db_handle = Prisma(use_dotenv=True, auto_register=True)

app = FastAPI(name="SparkUI - API")

logger: logging.Logger = None


@app.on_event("startup")
async def on_startup():
    global logger

    logger = logging.getLogger("uvicorn.info")
    generation_queue.start_queue()


async def init():
    await db_handle.connect()
