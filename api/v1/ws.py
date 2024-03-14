import io
import os

from dataclasses import dataclass

import prisma.models as pm

from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.responses import Response, StreamingResponse, FileResponse

from stable_diffusion.types import SDCheckpoint, Image
from stable_diffusion.generation_request import Txt2ImgRequest

from api_utils import image_to_response
from image_utils import merge_images

router = APIRouter(prefix="/v1", tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()


def init_routes(app: FastAPI):
    app.include_router(router)
