from enum import Enum
from dataclasses import dataclass


from hash_utils import get_sha256


class StableDiffusionBaseModel(Enum):
    SD1_5 = 1
    SD2_1 = 2
    SDXL1_0 = 10
    SDXL1_0Turbo = 20
    SDXL1_0Lightning = 30
