from typing import List, Optional

from pydantic import BaseModel

# Checkpoints
class CheckpointUsageInfo(BaseModel):
    width: Optional[int]
    height: Optional[int]
    clipSkip: float
    minSteps: Optional[int]
    maxSteps: Optional[int]
    sampler: Optional[str]

class CheckpointVariation(BaseModel):
    handle: str
    name: str
    preview_url: str
    usage_info: CheckpointUsageInfo
    
class Checkpoint(BaseModel):
    hanlde: str
    name: str
    variations: List[CheckpointVariation]

# Samplers
class Sampler(BaseModel):
    handle: str
    name: str
    group: str

# Post Bodies
class Post_Checkpoint(BaseModel):
    handle: str
    name: str
