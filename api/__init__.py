import os

from dotenv import load_dotenv
load_dotenv()

import api.v1.stable_diffusion
import api.v1.status

from fastapi import FastAPI, APIRouter

SPARKUI_BACK_HOST = os.getenv("SPARKUI_BACK_HOST")
SPARKUI_BACK_PORT = int(os.getenv("SPARKUI_BACK_PORT"))

app = FastAPI(title="SparkUI")

app.include_router(api.v1.stable_diffusion.init(), prefix = "/v1/stable_diffusion", tags = ["Stable Diffusion"])
app.include_router(api.v1.status.init(), prefix = "/v1/status", tags = ["System"])

def init():
	import uvicorn
	uvicorn.run(app, host = SPARKUI_BACK_HOST, port = SPARKUI_BACK_PORT)
