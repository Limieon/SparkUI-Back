from fastapi import APIRouter, UploadFile, Request
from typing import Union
from pydantic import BaseModel

router = APIRouter()

@router.get("/")
def get_status():
    return { "success": True }

def init():
    return router
