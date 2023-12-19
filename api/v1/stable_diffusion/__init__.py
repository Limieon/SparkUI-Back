from fastapi import APIRouter, UploadFile
from typing import Union, Optional, List

from api import app

from api.v1.schemas import Checkpoint

router = APIRouter()

# Checkpoint Endpoints
@router.get("/checkpoints")
def get_checkpoints() -> List[str]: 
	return []

@router.get("/checkpoints/{checkpoint}")
def get_checkpoint(checkpoint: str) -> Checkpoint:
    return {}

@router.post("/checkpoint")
def post_checkpoint():
    return {}

@router.put("/checkpoints/{checkpoint}")
def put_checkpoint(checkpoint: str):
    return {}

def init():
    return router
