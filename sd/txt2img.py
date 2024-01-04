import os
from os import path

import uuid

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from diffusers import (
    DDPMScheduler,
    UNet2DModel,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from PIL import Image

from sd.checkpoint_manager import request_checkpoint

from api.v1.schemas import (
    Txt2Img_GenerationRequest,
)
from prisma.models import (
    CheckpointVariation,
    Image,
    ImageGroup,
    GeneratedImage,
    GenerationData,
)

from config import SparkUIConfig as Config


from api.socket import sockets_broadcast, SocketMessageID


async def queue_txt2img(data: Txt2Img_GenerationRequest):
    return await generate_txt2img(data)


async def generate_txt2img(data: Txt2Img_GenerationRequest):
    checkpoint = await CheckpointVariation.prisma().find_first(
        where={
            "checkpointHandle": data.checkpoint.split("/")[0],
            "handle": data.checkpoint.split("/")[1],
        }
    )

    print(
        f"Generating {data.iterations} images with {data.steps} steps using {checkpoint.name}... ({data.outputWidth}x{data.outputHeight})"
    )

    print("Requesting checkpoint...")
    pipeline = await request_checkpoint(checkpoint.file, data.precision)
    print(pipeline.dtype)

    pipeline.safety_checker = None

    print("Creting Generator...")
    generator = torch.Generator("cuda").manual_seed(data.seed)

    print("Defining Scheduler...")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
    )

    print("Setting VAE...")
    pipeline.vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=pipeline.dtype
    ).to("cuda")

    print("Generating image...")
    image = pipeline(
        data.prompt,
        negative_prompt=data.negativePrompt,
        generator=generator,
        num_inference_steps=data.steps,
        width=data.outputWidth,
        height=data.outputHeight,
        clip_skip=2,
        guidance_scale=data.cfgScale,
    ).images[0]

    import utils

    image_id = await utils.store_txt2img_generated_image(image, data)

    await sockets_broadcast(
        SocketMessageID.on_image_generated,
        {"id": image_id},
    )
