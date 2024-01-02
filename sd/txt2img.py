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
from prisma.models import CheckpointVariation


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

    pipeline = await request_checkpoint(checkpoint.file, data.precision)
    generator = torch.Generator("cuda").manual_seed(data.seed)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
    ).to("cuda")

    image = pipeline(
        data.prompt,
        generator=generator,
        num_inference_steps=data.steps,
        width=data.outputWidth,
        height=data.outputHeight,
    ).images[0]

    image.save("image.png")
