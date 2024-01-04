import os
import os.path as path
import time

import requests
from urllib.parse import urlsplit
from fastapi.exceptions import HTTPException

import re

from tqdm import tqdm
from typing import List
from fastapi import UploadFile
from os.path import basename

from PIL import Image as PILImage

from config import SparkUIConfig as Config

from api.v1.schemas import (
    Txt2Img_GenData,
)

from prisma.models import Image, ImageGroup, GeneratedImage, Txt2Img_GenerationData


async def upload_file(dir: str, file: UploadFile):
    content = await file.read()
    with open(os.path.join(dir, file.filename), "wb") as f:
        f.write(content)


def download_file(url: str, dir: str, local_filename: str = None):
    filename = local_filename

    if filename == None:
        with requests.get(url, stream=True) as response:
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition and "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')
            else:
                filename = basename(urlsplit(url).path)

    with requests.get(url, stream=True) as response:
        total_size = int(response.headers.get("content-length", 0))
        with open(path.join(dir, filename), "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

    return path.join(dir, filename)


def get_handle_from_string(val: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", val)


def get_checkpoint_variation_handle(handle: str):
    return (handle.split("/")[0], handle.split("/")[1])


async def store_txt2img_generated_image(
    image: PILImage,
    data: Txt2Img_GenData,
    group_id: int = 1,
):
    time_ms = int(time.time())

    filename = os.path.join(
        Config.StableDiffusion.Directories.IMAGES_OUT, f"{time_ms}.png"
    )

    group = await ImageGroup.prisma().find_first(where={"id": group_id})
    if group == None:
        raise Exception(f"Cannot find image group with id {group_id}!")

    (checkpoint_handle, checkpoint_variation_handle) = get_checkpoint_variation_handle(
        data.checkpoint
    )

    print(f"{checkpoint_handle}-{checkpoint_variation_handle}")

    image_id = (
        await Image.prisma().create(
            data={
                "fileName": filename,
                "imageGroupId": group.id,
            }
        )
    ).id

    await GeneratedImage.prisma().create(
        data={
            "imageId": image_id,
            "generationDataId": (
                await Txt2Img_GenerationData.prisma().create(
                    data={
                        "checkpointHandle": checkpoint_handle,
                        "checkpointVariationHandle": checkpoint_variation_handle,
                        "sampler": data.sampler,
                        "precision": data.precision,
                        "vaeID": 1,
                        "vaeVersionId": 1,
                        "steps": data.steps,
                        "cfg_scale": data.cfgScale,
                        "seed": data.seed,
                        "width": data.outputWidth,
                        "height": data.outputHeight,
                        "prompt": data.prompt,
                        "negativePrompt": data.negativePrompt,
                        "stylePrompt": data.stylePrompt,
                        "negativeStylePrompt": data.negativePrompt,
                    }
                )
            ).id,
        }
    )

    image.save(filename)

    return image_id
