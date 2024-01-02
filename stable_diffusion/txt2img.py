from api.v1.schemas import (
    Txt2Img_GenerationRequest,
)


async def generate(data: Txt2Img_GenerationRequest):
    print(f"Generating {data.iterations} images with {data.steps} steps...")
    return {"success": True}
