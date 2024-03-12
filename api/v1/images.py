import io
import os

from dataclasses import dataclass

import prisma.models as pm

from fastapi import FastAPI, APIRouter
from fastapi.responses import Response, StreamingResponse, FileResponse

from stable_diffusion.types import SDCheckpoint, Image
from stable_diffusion.generation_request import Txt2ImgRequest

from api_utils import image_to_response
from image_utils import merge_images


router = APIRouter(prefix="/api/v1/images", tags=["Images"])


@router.get("/{id}")
async def get_image(id: int):
    img = await pm.Image.prisma().find_first(where={"id": id})
    if not img:
        return Response("Not found", status_code=404)

    return Image(
        id=img.id,
        width=img.width,
        height=img.height,
        created_at=int(img.createdAt.timestamp()),
        updated_at=int(img.updatedAt.timestamp()),
    )


@router.get("/{id}/full.png")
async def get_image_full(id: int):
    img = await pm.Image.prisma().find_first(where={"id": id})
    if not img:
        return Response("Not found", status_code=404)

    return FileResponse(img.file, media_type="image/jpg")


def init_routes(app: FastAPI):
    app.include_router(router)
