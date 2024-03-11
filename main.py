import asyncio
import os
import sys

from fastapi import FastAPI

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.importer import sd_import_models, SDImportConfig
from stable_diffusion.pipeline_manager import PipelineManager

from dotenv import load_dotenv

load_dotenv()


async def main():
    if "--import" in sys.argv:
        await import_models()

    await start_inference_server()


async def import_models():
    await sd_import_models(
        SDImportConfig(
            models_dir=os.getenv("SPARK_DIRS_MODELS"),
            checkpoints=os.getenv("SPARK_DIRS_SD_CHECKPOINT"),
            embeddings=os.getenv("SPARK_DIRS_SD_EMBEDDING"),
            loras=os.getenv("SPARK_DIRS_SD_LORAS"),
            lycorsis=os.getenv("SPARK_DIRS_SD_LYCORSIS"),
            control_nets=os.getenv("SPARK_DIRS_SD_CONTROL_NETS"),
            vaes=os.getenv("SPARK_DIRS_SD_VAES"),
        )
    )


async def start_inference_server():
    import api
    import spark

    await api.serve(spark.app, os.getenv("SPARK_API_HOST"), int(os.getenv("SPARK_API_PORT")))


if __name__ == "__main__":
    asyncio.run(main())
