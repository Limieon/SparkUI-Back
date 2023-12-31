from fastapi import UploadFile
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from datetime import datetime


# Checkpoints
class CheckpointUsageInfo(BaseModel):
    width: Optional[int]
    height: Optional[int]
    clip_skip: float
    min_steps: Optional[int]
    max_steps: Optional[int]
    sampler: Optional[str]


class CheckpointVariation(BaseModel):
    handle: str
    name: str
    preview_url: str
    usage_info: CheckpointUsageInfo
    created_at: int


class Checkpoint(BaseModel):
    name: str
    variations: List[CheckpointVariation]
    created_at: int
    last_updated: int


# Samplers
class Sampler(BaseModel):
    handle: str
    name: str
    group: str


# Request Bodies
class Post_Checkpoint(BaseModel):
    handle: str
    name: str


class Post_CheckpointVariation(BaseModel):
    handle: str
    name: str
    base_model: str
    preview_url: str
