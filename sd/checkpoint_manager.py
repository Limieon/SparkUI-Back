import os
import time
from collections import OrderedDict

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from diffusers import (
    DDPMScheduler,
    UNet2DModel,
    DiffusionPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
import torch

from config import SparkUIConfig as Config


@dataclass
class CachedPipeline:
    pipeline: StableDiffusionPipeline
    id: int
    precision: torch.dtype


cached_checkpoints = OrderedDict[str, CachedPipeline]()
last_id = 0


async def request_checkpoint(path: str, precision: str) -> StableDiffusionPipeline:
    global last_id
    dtype = torch.float32 if precision == "fp32" else torch.float16

    if path in cached_checkpoints.keys():
        item = cached_checkpoints[path]
        if item.precision != dtype:
            del cached_checkpoints[path]
            return await request_checkpoint(path, precision)

        print("Using cached checkpoint...")
        return item.pipeline

    if len(cached_checkpoints) > Config.StableDiffusion.MAX_LOADED_CHECKPOINTS:
        cached_checkpoints.popitem(last=False)

    pipeline = StableDiffusionPipeline.from_single_file(
        path, use_safetensors=True, dtype=dtype
    ).to("cuda")

    print(last_id)
    last_id = last_id + 1
    cached_checkpoints[path] = CachedPipeline(
        pipeline=pipeline, id=last_id, precision=dtype
    )

    return pipeline
