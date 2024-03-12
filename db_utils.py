import os
import requests

from image_utils import download_image

import prisma.models as pm
from PIL.Image import Image


async def add_image_by_url(url: str, path: str) -> int:
    try:
        os.makedirs(path)
    except:
        print("Directory already exists!")

    file, width, height = await download_image(url, path)

    return await pm.Image.prisma().create(
        {
            "file": file,
            "format": file.split(".")[-1],
            "width": width,
            "height": height,
        }
    )
