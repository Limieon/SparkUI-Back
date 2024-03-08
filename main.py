import asyncio

from fastapi import FastAPI

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.pipeline_manager import PipelineManager


async def main():
    import api
    import spark

    await api.serve(spark.app, "0.0.0.0", 1911)


if __name__ == "__main__":
    asyncio.run(main())
