import io
from fastapi.responses import StreamingResponse

from PIL.Image import Image


def image_to_response(image: Image, format: str = "png", status_code: int = 200) -> StreamingResponse:
    stream = io.BytesIO()
    image.save(stream, format=format)
    stream.seek(0)

    return StreamingResponse(io.BytesIO(stream.read()), media_type=f"image/{format}", status_code=status_code)
