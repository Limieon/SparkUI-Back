import asyncio

from fastapi import FastAPI

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.pipeline_manager import PipelineManager

from config import SparkConfig


async def main():
    import api
    import spark

    await api.serve(spark.app, SparkConfig.API.host, SparkConfig.API.port)


if __name__ == "__main__":
    asyncio.run(main())
