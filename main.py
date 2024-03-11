import asyncio
import os
import sys

from fastapi import FastAPI

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.pipeline_manager import PipelineManager

from dotenv import load_dotenv

load_dotenv()


async def main():
    if "--import" in sys.argv:
        await import_models()

    await start_inference_server()


async def import_models():
    print("Importing new models...")


async def start_inference_server():
    import api
    import spark

    await api.serve(spark.app, os.getenv("SPARK_API_HOST"), int(os.getenv("SPARK_API_PORT")))


if __name__ == "__main__":
    asyncio.run(main())
