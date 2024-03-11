import asyncio
import os

from fastapi import FastAPI

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.pipeline_manager import PipelineManager

from dotenv import load_dotenv

load_dotenv()


async def main():
    import api
    import spark

    await api.serve(spark.app, os.getenv("SPARK_API_HOST"), int(os.getenv("SPARK_API_PORT")))


if __name__ == "__main__":
    asyncio.run(main())
