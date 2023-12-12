from typing import Union

from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
	return { "success": True }


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host = "0.0.0.0", port = 20177)
