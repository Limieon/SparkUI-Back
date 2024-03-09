from pydantic import BaseModel


class LoraRequest(BaseModel):
    lora: str = "lora_handle"
    weight: float = 1.0


class Txt2ImgRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""

    width: int = 512
    height: int = 512
    cfg_scale: float = 7.0
    steps: int = 20

    loras: list[LoraRequest]

    num_images: int = 1
