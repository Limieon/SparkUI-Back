from fastapi import UploadFile

from config import SparkUIConfig as Config
from utils import upload_file

async def upload_checkpoint(file: UploadFile):
    await upload_file(Config.Directories.StableDiffusion.CHECKPOINT, file.filename)
