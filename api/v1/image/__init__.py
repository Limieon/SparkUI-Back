import os

from fastapi import APIRouter, UploadFile, Response
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse

from typing import Union, Optional, List

from main import db
from prisma.models import (
    Checkpoint,
    CheckpointVariation,
    Image,
    GeneratedImage,
    Txt2Img_GenerationData,
)
from prisma.errors import UniqueViolationError

from config import SparkUIConfig as Config
from sd.checkpoint import upload_checkpoint
from api.v1.schemas import (
    Image_Response,
    Images_Response,
    Txt2Img_GenData,
    Prompt_Response,
)

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
                url_full=f"/v1/image/{i.id}/full.png",
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


@router.get("/{id}/prompt")
async def get_image(id: int) -> Prompt_Response:
    try:
        gen_data = await Txt2Img_GenerationData.prisma().find_first(
            where={
                "id": (
                    await GeneratedImage.prisma().find_first(where={"imageId": id})
                ).id
            }
        )

        return Prompt_Response(
            positive=gen_data.prompt,
            negative=gen_data.negativePrompt,
            positiveStyle=gen_data.stylePrompt,
            negativeStyle=gen_data.negativeStylePrompt,
        )
    except:
        raise HTTPException(status_code=404)


@router.get("/{id}/seed")
async def get_image(id: int) -> int:
    try:
        gen_data = await Txt2Img_GenerationData.prisma().find_first(
            where={
                "id": (
                    await GeneratedImage.prisma().find_first(where={"imageId": id})
                ).id
            }
        )

        return gen_data.seed
    except:
        raise HTTPException(status_code=404)


@router.get("/{id}/gen_data")
async def get_image(id: int) -> Txt2Img_GenData:
    try:
        gen_data = await Txt2Img_GenerationData.prisma().find_first(
            where={
                "id": (
                    await GeneratedImage.prisma().find_first(where={"imageId": id})
                ).id
            }
        )

        return Txt2Img_GenData(
            prompt=gen_data.prompt,
            negativePrompt=gen_data.negativePrompt,
            stylePrompt=gen_data.stylePrompt,
            negativeStylePrompt=gen_data.negativeStylePrompt,
            checkpoint=f"{gen_data.checkpointHandle}/{gen_data.checkpointVariationHandle}",
            steps=gen_data.steps,
            cfgScale=gen_data.cfg_scale,
            seed=gen_data.seed,
            outputWidth=gen_data.width,
            outputHeight=gen_data.height,
            precision=gen_data.precision,
            vae="default/default",
            loras=[],
            sampler=gen_data.sampler,
            iterations=1,  # Will be removed in the future
        )
    except:
        raise HTTPException(status_code=404)


@router.get("/{id}/meta")
async def get_image_meta():
    return {"success": True}


def init():
    return router
