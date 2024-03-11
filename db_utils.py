import os
import requests

from image_utils import download_image

from prisma.models import Image


async def add_image_by_url(url: str, path: str) -> int:
    try:
        os.makedirs(path)
    except:
        print("Directory already exists!")

    file = await download_image(url, path)

    return await Image.prisma().create(
        {
            "file": file,
            "format": file.split(".")[-1],
        }
    )
