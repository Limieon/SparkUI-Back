import os

from fastapi import APIRouter, UploadFile, Response
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse

from typing import Union, Optional, List

from main import db
from prisma.models import Checkpoint, CheckpointVariation, Image
from prisma.errors import UniqueViolationError

from config import SparkUIConfig as Config
from sd.checkpoint import upload_checkpoint
from api.v1.schemas import Image_Response, Images_Response

from api.socket import sockets_broadcast

router = APIRouter()


@router.get("/")
async def get_images(offset: int = 0, limit: int = 50) -> Images_Response:
    res: list[Image_Response] = []
    for i in await Image.prisma().find_many(take=limit, skip=offset):
        res.append(
            Image_Response(
                file_name=i.fileName,
                created_at=int(i.created_at.timestamp()),
                url_full=f"/v1/image/{id}/full.png",
            )
        )

    return Images_Response(
        images=res, amount=len(res), available=await Image.prisma().count()
    )


@router.get(
    "/{id}/full.png",
    response_class=Response,
    responses={
        200: {
            "description": "Return the full-size image",
            "content": {"image/png": {}},
        },
        404: {"description": "Image not found"},
    },
)
async def get_image_full(id: int) -> FileResponse:
    try:
        image = await Image.prisma().find_first(where={"id": id})
        return FileResponse(image.fileName)
    except:
        raise HTTPException(status_code=404)


@router.get("/{id}")
async def get_image(id: int) -> Image_Response:
    try:
        image = await Image.prisma().find_first(where={"id": id})

        return Image_Response(
            file_name=image.fileName,
            created_at=int(image.created_at.timestamp()),
            url_full=f"/v1/image/{id}/full.png",
        )
    except:
        raise HTTPException(status_code=404)


def init():
    return router
