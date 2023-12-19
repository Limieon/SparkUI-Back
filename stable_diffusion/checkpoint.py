import os

from fastapi import UploadFile

from config import SparkUIConfig as Config

async def upload_checkpoint(file: UploadFile):
    content = await file.read()
    with open(os.path.join(Config.Directories.StableDiffusion.CHECKPOINT, file.filename), 'wb') as f:
        f.write(content)
