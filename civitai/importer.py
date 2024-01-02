import os
import os.path as path
from urllib.parse import urlsplit
from tqdm import tqdm

import requests
import json
import re

from dataclasses import dataclass

from utils import get_handle_from_string, download_file

from typing import List
from fastapi import HTTPException
from datetime import datetime

from prisma.models import Checkpoint, CheckpointVariation

from config import SparkUIConfig as Config

from api.socket import sockets_broadcast, SocketMessageID

CIVITAI_BASE_URL = "https://civitai.com/api/v1"


@dataclass
class ImportRequest:
    base_id: int
    variation_id: int
    handle: str
    name: str
    preview_url: int
    download_url: str
    base_model: str
    base_handle: str


importer_queue: list[ImportRequest] = []


async def import_models(base_id: int, ids: List[int]):
    """Downloads models from CivitAI and pets them into the database

    Args:
        baseid (int): the base model id
        ids (List[int]): the variation ids to download

    Raises:
        HTTPException: Gets thrown when an error occurs
    """

    base = requests.get(f"{CIVITAI_BASE_URL}/models/{base_id}")
    if base.status_code != 200:
        raise HTTPException(
            500,
            f"Could not fetch data from CivitAI! ({base.status_code}), GET {CIVITAI_BASE_URL}/models/{base_id}",
        )

    base_data = base.json()
    base_data_name: str = base_data["name"]
    base_data_handle = get_handle_from_string(base_data_name)

    if not (await Checkpoint.prisma().find_first(where={"handle": base_data_handle})):
        await Checkpoint.prisma().create(
            {
                "handle": base_data_handle,
                "name": base_data_name,
                "civitai_id": base_id,
                "created_at": datetime.now(),
                "last_updated": datetime.now(),
            }
        )

    for id in ids:
        for variation_data in base_data["modelVersions"]:
            if variation_data["id"] != id:
                continue

            variation_data_name: str = variation_data["name"]
            importer_queue.append(
                ImportRequest(
                    base_id=base_id,
                    variation_id=id,
                    handle=get_handle_from_string(variation_data_name),
                    name=f"{base_data_name} - {variation_data_name}",
                    base_model=get_handle_from_string(variation_data["baseModel"]),
                    download_url=variation_data["downloadUrl"],
                    preview_url=variation_data["images"][0]["url"],
                    base_handle=base_data_handle,
                )
            )


async def download_file(data: ImportRequest, dir: str, local_filename: str = None):
    filename = local_filename

    if filename == None:
        with requests.get(url=data.download_url, stream=True) as response:
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition and "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')
            else:
                filename = path.basename(urlsplit(url).path)

    with requests.get(data.download_url, stream=True) as response:
        total_size = int(response.headers.get("content-length", 0))
        current = 0
        with open(path.join(dir, filename), "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            last_sent = -999999999
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    chunk_size = len(chunk)

                    # bar.update(chunk_size)
                    current = current + chunk_size

                    print(current - last_sent)
                    if (current - last_sent) >= 5242880:  # 5242880 = 5MB
                        last_sent = current
                        socket_data = {
                            "current": current,
                            "size": total_size,
                            "name": data.name,
                            "preview_url": data.preview_url,
                            "handle": data.handle,
                            "queue": [],
                        }

                        for val in importer_queue:
                            socket_data["queue"].append(
                                {
                                    "name": val.name,
                                    "preview_url": val.preview_url,
                                    "handle": val.handle,
                                }
                            )

                        await sockets_broadcast(
                            SocketMessageID.civitai_importer_update, socket_data
                        )

    return path.join(dir, filename)


async def import_model(data: ImportRequest):
    await CheckpointVariation.prisma().create(
        {
            "handle": data.handle,
            "name": data.name,
            "baseModel": data.base_model,
            "file": await download_file(
                data,
                Config.StableDiffusion.Directories.CHECKPOINT,
            ),
            "previewUrl": data.preview_url,
            "civitai_id": data.variation_id,
            "checkpointHandle": data.base_handle,
            "created_at": datetime.now(),
        }
    )


async def importer_queue_step():
    if len(importer_queue) < 1:
        return

    await import_model(importer_queue.pop())
    await sockets_broadcast(SocketMessageID.civitai_importer_update, {})
