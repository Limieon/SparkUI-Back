import io
import os

from dataclasses import dataclass

import prisma.models as pm

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from stable_diffusion.types import SDCheckpoint, Image
from stable_diffusion.generation_request import Txt2ImgRequest
from stable_diffusion.workflow import WorkflowData, Workflow

from spark import generation_queue

from api_utils import image_to_response
from image_utils import merge_images

router = APIRouter(prefix="/api/v1/stable_diffusion", tags=["StableDiffusion"])


@router.post("/invoke")
async def post_workflow(data: WorkflowData, client_id: str):
    if client_id is None:
        raise HTTPException(403, "No clientID provided!")

    workflow = Workflow(nodes=data.nodes)

    if workflow.has_recursion():
        raise HTTPException(400, "Workflow contains recursion!")

    await workflow.invoke(parameters=data.parameters)

    return {}


@router.get("/nodes")
async def post_workflow(data: WorkflowData, client_id: str):
    # TODO implement node endpoint

    return {}


def init_routes(app: FastAPI):
    app.include_router(router)
