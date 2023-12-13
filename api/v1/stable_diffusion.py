import os

from fastapi import APIRouter, UploadFile, Request, File
from typing import Union
from pydantic import BaseModel

from db import Checkpoint, CheckpointVariation, db

from main import SD_CHECKPOINT_DIR

router = APIRouter()

# Upload Endpoints
@router.post("/checkpoints")
def create_checkpoint_group(handle: str, name: str, description: str, preview_url: str, civitai_page: str):
    Checkpoint.create(handle = handle, displayName = name, description = description, preview_url = preview_url, civitai_page = civitai_page)
    return { "success": True }

@router.put("/checkpoints/{group}")
async def create_checkpoint_group(group: str, handle: str, name: str, checkpoint: UploadFile):
    existing_checkpoint = Checkpoint.get(Checkpoint.handle == group)
    if not existing_checkpoint:
        raise Exception(f"Checkpoint group '{group}' does not exist!")
    
    file_content = await checkpoint.read()
    with open(os.path.join(SD_CHECKPOINT_DIR, checkpoint.filename), 'wb') as f:
        f.write(file_content)
    
    CheckpointVariation.create(
        checkpoint=existing_checkpoint,
        handle=handle,
        name=name,
        file_name=checkpoint.filename,
        inpainting=False
    )
    
    return { "success": True }

# Interference Endpoints

# Query Endpoints
@router.get("/checkpoints")
def get_checkpoints():
    handles = []
    for checkpoint in Checkpoint.select():
        handles.append(checkpoint.handle)
        
    return handles

@router.get("/variants")
def get_variants(checkpoint: Union[str, None]):
    pass
        
    return handles
@router.get("/checkpoints/{group}/{variant}")
def get_checkpoints(group: str, variant: str):
    return { "success": True }

def init():
    return router
