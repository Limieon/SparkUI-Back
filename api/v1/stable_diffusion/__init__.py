import io
import os

from fastapi import FastAPI, APIRouter
from fastapi.responses import Response, StreamingResponse

from stable_diffusion import StableDiffusionBaseModel
from stable_diffusion.generation_request import Txt2ImgRequest

from spark import pipeline_manager

from PIL.Image import Image

from api_utils import image_to_response
from image_utils import merge_images

router = APIRouter(prefix="/api/v1/stable_diffusion", tags=["StableDiffusion"])


@router.post("/generate/{model}")
async def generate(model: str, data: Txt2ImgRequest):
    model_path = os.path.join("./assets/models/StableDiffusion/", model)
    pipe = pipeline_manager.load_pipeline(model_path, StableDiffusionBaseModel.SDXL1_0, use_gpu=True)

    for lora in data.loras:
        pipe.load_lora_weights("./assets/models/Lora", weight_name=lora.lora, weight=lora.weight)

    return image_to_response(
        merge_images(
            pipe(
                prompt=data.prompt,
                negative_prompt=data.negative_prompt,
                num_inference_steps=data.steps,
                num_images_per_prompt=data.num_images,
                guidance_scale=data.cfg_scale,
                width=data.width,
                height=data.height,
            ).images
        )
    )


def init_routes(app: FastAPI):
    app.include_router(router)
