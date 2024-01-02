import os

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


async def request_checkpoint(path: str, precision: str) -> StableDiffusionPipeline:
    dtype = torch.float32 if precision == "fp32" else torch.float16

    return StableDiffusionPipeline.from_single_file(
        path, use_safetensors=True, torch_dtype=dtype
    ).to("cuda")
