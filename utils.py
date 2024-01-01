import os
import os.path as path

import requests
from urllib.parse import urlsplit
from fastapi.exceptions import HTTPException

import re

from tqdm import tqdm
from typing import List
from fastapi import UploadFile
from os.path import basename

from config import SparkUIConfig as Config


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
