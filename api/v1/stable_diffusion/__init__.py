import os

from fastapi import APIRouter, UploadFile
from fastapi.exceptions import HTTPException

from typing import Union, Optional, List

from main import db
from prisma.models import Checkpoint, CheckpointVariation
from prisma.errors import UniqueViolationError

from config import SparkUIConfig as Config
from stable_diffusion.checkpoint import upload_checkpoint
from api.v1.schemas import (
    Checkpoint as S_Checkpoint,
    CheckpointVariation as S_CheckpointVariation,
    CheckpointUsageInfo as S_CheckpointUsageInfo,
    Sampler,
    Post_Checkpoint,
    Post_CheckpointVariation,
)

from api.socket import broadcast

import civitai.importer

router = APIRouter()


# Checkpoints
@router.get("/checkpoints", tags=["Checkpoint"])
async def get_checkpoints() -> List[str]:
    data = await Checkpoint.prisma().find_many()
    handles: List[str] = []
    for d in data:
        handles.append(d.handle)

    return handles


@router.get("/checkpoints/{checkpoint}", tags=["Checkpoint"])
async def get_checkpoints(checkpoint: str) -> S_Checkpoint:
    db_checkpoint = await Checkpoint.prisma().find_first(where={"handle": checkpoint})

    variations: list[S_CheckpointVariation] = []
    for d in await CheckpointVariation.prisma().find_many(
        where={"checkpointHandle": checkpoint}
    ):
        variations.append(
            S_CheckpointVariation(
                handle=f"{checkpoint}/{d.handle}",
                name=d.name,
                preview_url=d.previewUrl,
                base_model=d.baseModel,
                created_at=int(d.created_at.timestamp()),
                usage_info=S_CheckpointUsageInfo(
                    width=None,
                    height=None,
                    clip_skip=1,
                    min_steps=None,
                    max_steps=None,
                    sampler=None,
                ),
            )
        )

    return S_Checkpoint(
        name=db_checkpoint.name,
        created_at=int(db_checkpoint.created_at.timestamp()),
        last_updated=int(db_checkpoint.last_updated.timestamp()),
        variations=variations,
    )


@router.post("/checkpoints", tags=["Checkpoint"])
async def post_checkpoints(body: Post_Checkpoint):
    try:
        await Checkpoint.prisma().create(
            data={"handle": body.handle, "name": body.name}
        )
    except:
        raise HTTPException(400, f"Handle '{body.handle}' is already in use!")

    return {"success": True}


@router.post("/checkpoints/{checkpoint}", tags=["Checkpoint"])
async def post_checkpoints(checkpoint: str, body: Post_CheckpointVariation):
    base_found = False

    for k, v in vars(Config.StableDiffusion.BaseModels).items():
        if k.startswith("__"):
            continue
        if v.handle == body.base_model:
            base_found = True
            break

    if not base_found:
        raise HTTPException(400, f"Base model '{body.base_model}' is not valid!")

    db_checkpoint = await Checkpoint.prisma().find_first(
        where={"handle": {"equals": checkpoint}}
    )

    if db_checkpoint is None:
        raise HTTPException(400, f"Checkpoint group '{checkpoint}' not found!")

    try:
        await CheckpointVariation.prisma().create(
            data={
                "handle": body.handle,
                "name": body.name,
                "checkpointHandle": checkpoint,
                "baseModel": body.base_model,
                "previewUrl": body.preview_url,
            }
        )

    except UniqueViolationError:
        raise HTTPException(400, f"Handle '{body.handle}' is already in use!")

    return {"success": True}


@router.put("/checkpoints/{checkpoint}/{variation}", tags=["Checkpoint"])
async def put_checkpoints(checkpoint: str, variation: str, file: UploadFile):
    db_variation = await CheckpointVariation.prisma().find_first(
        where={
            "AND": [
                {"handle": {"equals": variation}},
                {"checkpointHandle": {"equals": checkpoint}},
            ]
        }
    )

    if db_variation is None:
        raise HTTPException(
            400,
            f"Checkpoint variation '{variation}' not found in group '{checkpoint}'!",
        )

    await db_variation.prisma().update(
        data={"file": file.filename},
        where={"handle": variation, "checkpointHandle": checkpoint},
    )

    await upload_checkpoint(file)

    return {"success": True}


@router.delete("/checkpoints/{checkpoint}/{variation}", tags=["Checkpoint"])
async def delete_checkpoint_variation(checkpoint: str, variation: str):
    data = await CheckpointVariation.prisma().delete(
        where={"handle": variation, "checkpointHandle": checkpoint}
    )

    print(data.file)
    os.remove(data.file)

    return {"success": True}


# Loras
@router.get("/loras", tags=["LoRA"])
async def get_loras():
    await broadcast("lol")
    return []


@router.get("/loras/{lora}", tags=["LoRA"])
def get_loras(lora: str):
    return {}


@router.post("/loras", tags=["LoRA"])
def post_loras():
    return {}


@router.put("/loras/{lora}", tags=["LoRA"])
def put_loras(lora: str):
    return {}


# Embeddings
@router.get("/embeddings", tags=["Embedding"])
def get_embeddings():
    return []


@router.get("/embeddings/{embedding}", tags=["Embedding"])
def get_embeddings(embedding: str):
    return {}


@router.post("/embeddings", tags=["Embedding"])
def post_embeddings():
    return {}


@router.put("/embeddings/{embedding}", tags=["Embedding"])
def put_embeddings(embedding: str):
    return {}


# VAEs
@router.get("/vaes", tags=["VAE"])
def get_vaes():
    return []


@router.get("/vaes/{vae}", tags=["VAE"])
def get_vaes(vae: str):
    return {}


@router.post("/vaes", tags=["VAE"])
def post_vaes():
    return {}


@router.put("/vaes/{vae}", tags=["VAE"])
def put_vaes(vae: str):
    return {}


# Samplers
@router.get("/samplers", tags=["Sampler"])
def get_samplers() -> List[Sampler]:
    return []


# CivitAI
@router.post("/civitai/import/{modelid}", tags=["CivitAI"])
async def post_civitai_import(modelid: int, versionIDs: List[int]):
    try:
        await civitai.importer.import_models(modelid, versionIDs)
    except:
        raise

    return {"success": True}


def init():
    return router
