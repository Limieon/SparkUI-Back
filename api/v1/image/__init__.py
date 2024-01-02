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
    Checkpoint as S_Checkpoint,
    CheckpointVariation as S_CheckpointVariation,
    CheckpointUsageInfo as S_CheckpointUsageInfo,
    Sampler,
    Post_Checkpoint,
    Post_CheckpointVariation,
    Txt2Img_GenerationRequest,
)

from api.socket import sockets_broadcast

import civitai.importer

router = APIRouter()


@router.get("/{id}/full")
async def get_image_full(id: int):
    image = await Image.prisma().find_first(where={"id": id})
    return FileResponse(image.fileName)


def init():
    return router
