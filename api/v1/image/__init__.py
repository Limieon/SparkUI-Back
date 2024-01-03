import os

from fastapi import APIRouter, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse

from typing import Union, Optional, List

from main import db
from prisma.models import Checkpoint, CheckpointVariation, Image
from prisma.errors import UniqueViolationError

from config import SparkUIConfig as Config
from sd.checkpoint import upload_checkpoint
from api.v1.schemas import (
    Image_Response,
)

from api.socket import sockets_broadcast

import civitai.importer

router = APIRouter()


@router.get("/{id}/full.png")
async def get_image_full(id: int):
    image = await Image.prisma().find_first(where={"id": id})
    return FileResponse(image.fileName)


@router.get("/{id}")
async def get_image(id: int):
    image = await Image.prisma().find_first(where={"id": id})

    return Image_Response(
        file_name=image.fileName,
        created_at=int(image.created_at.timestamp()),
        url_full=f"/v1/image/{id}/full.png",
    )


def init():
    return router
