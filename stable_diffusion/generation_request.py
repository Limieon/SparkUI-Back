from pydantic import BaseModel


class LoraRequest(BaseModel):
    def __hash__(self):
        return hash((self.lora, self.weight))

    lora: str = "lora_handle"
    weight: float = 1.0


class Txt2ImgRequest(BaseModel):
    def __hash__(self):
        return hash(
            (self.checkpoint, self.prompt, self.negative_prompt, self.width, self.height, self.cfg_scale, self.steps, tuple(self.loras), self.num_images)
        )

    checkpoint: str = ""
    prompt: str = ""
    negative_prompt: str = ""

    width: int = 512
    height: int = 512
    cfg_scale: float = 7.0
    steps: int = 20

    loras: list[LoraRequest]

    num_images: int = 1
