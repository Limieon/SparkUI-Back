import io
import os

from dataclasses import dataclass

import prisma.models as pm

from fastapi import FastAPI, APIRouter
from fastapi.responses import Response, StreamingResponse

from stable_diffusion.types import SDCheckpoint, Image
from stable_diffusion.generation_request import Txt2ImgRequest

from spark import generation_queue

from api_utils import image_to_response
from image_utils import merge_images

router = APIRouter(prefix="/api/v1/stable_diffusion", tags=["StableDiffusion"])


@router.post("/generate")
async def generate(data: Txt2ImgRequest):
    images = await generation_queue.queue_txt2img(data)
    return image_to_response(merge_images(images))


@dataclass
class Res_Checkpoints:
    items: list[SDCheckpoint]
    total: int
    filtered: int


@router.get("/checkpoints")
async def get_models():
    checkpoints: list[SDCheckpoint] = []

    for ckpt in await pm.StableDiffusionCheckpoint.prisma().find_many(include={"base": {}}):
        images: list[Image] = []

        for img in await pm.Image.prisma().find_many(where={"stableDiffusionBaseId": ckpt.baseID}):
            images.append(
                Image(id=img.id, width=img.width, height=img.height, created_at=int(img.createdAt.timestamp()), updated_at=int(img.updatedAt.timestamp()))
            )

        checkpoints.append(
            SDCheckpoint(
                id=ckpt.baseID,
                name=ckpt.base.name,
                description=ckpt.base.description,
                format=ckpt.base.format,
                sha256=ckpt.base.sha256,
                sd_base_model=ckpt.base.sdBaseModel,
                created_at=int(ckpt.base.createdAt.timestamp()),
                updated_at=int(ckpt.base.updatedAt.timestamp()),
                tags=ckpt.base.tags,
                preview_images=images,
            )
        )

    return Res_Checkpoints(
        items=checkpoints, total=await pm.StableDiffusionCheckpoint.prisma().count(), filtered=await pm.StableDiffusionCheckpoint.prisma().count()
    )


def init_routes(app: FastAPI):
    app.include_router(router)
