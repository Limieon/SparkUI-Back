import os
import requests
import json
import re

from utils import get_handle_from_string, download_file

from typing import List
from fastapi import HTTPException
from datetime import datetime

from prisma.models import Checkpoint, CheckpointVariation

from config import SparkUIConfig as Config

CIVITAI_BASE_URL = "https://civitai.com/api/v1"


async def import_models(baseid: int, ids: List[int]):
    """Downloads models from CivitAI and pets them into the database

    Args:
        baseid (int): the base model id
        ids (List[int]): the variation ids to download

    Raises:
        HTTPException: Gets thrown when an error occurs
    """

    base = requests.get(f"{CIVITAI_BASE_URL}/models/{baseid}")
    if base.status_code != 200:
        raise HTTPException(
            500,
            f"Could not fetch data from CivitAI! ({base.status_code}), GET {CIVITAI_BASE_URL}/models/{baseid}",
        )

    base_data = base.json()
    base_data_name: str = base_data["name"]
    base_data_handle = get_handle_from_string(base_data_name)

    if not (await Checkpoint.prisma().find_first(where={"handle": base_data_handle})):
        await Checkpoint.prisma().create(
            {"handle": base_data_handle, "name": base_data_name, "civitai_id": baseid, "created_at": datetime.now(), "last_updated": datetime.now()}
        )

    for id in ids:
        for variation_data in base_data["modelVersions"]:
            if variation_data["id"] != id:
                continue

            variation_data_name: str = variation_data["name"]
            await CheckpointVariation.prisma().create(
                {
                    "handle": get_handle_from_string(variation_data_name),
                    "name": f"{base_data_name} - {variation_data_name}",
                    "baseModel": get_handle_from_string(variation_data["baseModel"]),
                    "file": download_file(
                        variation_data["downloadUrl"],
                        Config.StableDiffusion.Directories.CHECKPOINT,
                    ),
                    "previewUrl": variation_data["images"][0]["url"],
                    "civitai_id": id,
                    "checkpointHandle": base_data_handle,
                    "created_at": datetime.now()
                }
            )
    
    await Checkpoint.prisma().update(
            where = { "handle": base_data_handle },
            data= {"last_updated": datetime.now()}
        )
