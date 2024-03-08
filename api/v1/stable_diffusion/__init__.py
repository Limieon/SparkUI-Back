import io

from fastapi import FastAPI, APIRouter
from fastapi.responses import Response, StreamingResponse

from hash_utils import encode_image_to_base64

router = APIRouter(prefix="/api/v1/stable_diffusion")

from stable_diffusion import StableDiffusionBaseModel
from spark import pipeline_manager

from PIL.Image import Image


from api_utils import image_to_response


@router.get("/generate")
async def generate(
    model: str,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    cfg_scale: float,
):
    pipe = pipeline_manager.load_pipeline(model, base=StableDiffusionBaseModel.SDXL1_0, use_gpu=True)
    image: Image = pipe(prompt=prompt, num_inference_steps=steps, width=width, height=height, guidance_scale=cfg_scale).images[0]

    image.save("test.png")

    return image_to_response(image)


def init_routes(app: FastAPI):
    app.include_router(router)
