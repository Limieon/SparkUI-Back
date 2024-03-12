from pydantic import BaseModel


class Image(BaseModel):
    id: int
    width: int
    height: int
    created_at: int
    updated_at: int


class SDCheckpoint(BaseModel):
    id: int
    name: str
    description: str
    format: str
    sha256: str
    sd_base_model: str

    created_at: int
    updated_at: int

    tags: list[str]

    preview_images: list[Image] = []
