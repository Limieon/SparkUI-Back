from fastapi import APIRouter, UploadFile, Request
from typing import Union
from pydantic import BaseModel

router = APIRouter()

@router.post("/checkpoints")
def create_checkpoint_group(handle: str, name: str, description: str, civitai_url: str, preview_image: str):
    return { "success": True }

@router.post("/checkpoints/{group}")
def create_checkpoint_group(group: str, handle: str, name: str, description: str, preview_image: str, checkpoint: UploadFile):    
    return { "success": True }

def init():
    return router
