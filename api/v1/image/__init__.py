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
from api.v1.schemas import Image_Response, Images_Response, Txt2Img_GenData

from api.socket import sockets_broadcast

router = APIRouter()


@router.get("/")
async def get_images(offset: int = 0, limit: int = 50) -> Images_Response:
    res: list[Image_Response] = []
    for i in await Image.prisma().find_many(take=limit, skip=offset):
        generation_data: Txt2Img_GenData = None

        generated_image = await GeneratedImage.prisma().find_first(where={"id": i.id})

        if generated_image:
            print(generated_image.generationDataId)
            db_gen_data = await Txt2Img_GenerationData.prisma().find_first_or_raise(
                where={"id": generated_image.generationDataId}
            )

            generation_data = Txt2Img_GenData(
                prompt=db_gen_data.prompt,
                negativePrompt=db_gen_data.negativePrompt,
                stylePrompt=db_gen_data.stylePrompt,
                negativeStylePrompt=db_gen_data.negativeStylePrompt,
                checkpoint=f"{db_gen_data.checkpointHandle}/{db_gen_data.checkpointVariationHandle}",
                steps=db_gen_data.steps,
                cfgScale=db_gen_data.cfg_scale,
                seed=db_gen_data.seed,
                outputWidth=db_gen_data.width,
                outputHeight=db_gen_data.height,
                precision=db_gen_data.precision,
                vae="default",
                vae_version="default",
                loras=[],
                sampler=db_gen_data.sampler,
                iterations=1,  # Will be removed in the future
            )

        res.append(
            Image_Response(
                file_name=i.fileName,
                created_at=int(i.created_at.timestamp()),
                url_full=f"/v1/image/{i.id}/full.png",
                generation_data=generation_data,
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


@router.get("/{id}/meta")
async def get_image_meta():
    return {"success": True}


def init():
    return router
