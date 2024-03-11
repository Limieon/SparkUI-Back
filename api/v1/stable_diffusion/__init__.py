import io
import os

from fastapi import FastAPI, APIRouter
from fastapi.responses import Response, StreamingResponse

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.generation_request import Txt2ImgRequest

from spark import generation_queue

from PIL.Image import Image

from api_utils import image_to_response
from image_utils import merge_images

router = APIRouter(prefix="/api/v1/stable_diffusion", tags=["StableDiffusion"])


@router.post("/generate")
async def generate(data: Txt2ImgRequest):
    images = await generation_queue.queue_txt2img(data)
    return image_to_response(merge_images(images))


def init_routes(app: FastAPI):
    app.include_router(router)
