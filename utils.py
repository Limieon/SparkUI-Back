import os

from fastapi import UploadFile

from config import SparkUIConfig as Config

async def upload_file(dir: str, file: UploadFile):
    content = await file.read()
    with open(os.path.join(dir, file.filename), 'wb') as f:
        f.write(content)
