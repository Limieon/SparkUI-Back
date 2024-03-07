from enum import Enum

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline

from hash_utils import gen_sha256

class StableDiffusionBaseVersion(Enum):
    SD1_5 = 1
    SD2_1 = 2
    SDXL1_0 = 10
    SDXLTurbo = 20
    SDXLLightning = 30

class StableDiffusionQueue:
    def __init__(self):
        self.pipelines = {}
        self.model_hashes = {}

        pass

    def set_max_models_in_vram(self, amount: int):
        self.max_models_in_vram = amount

    def get_hash_from_file(self, path: str):
        if not path in self.model_hashes:
            self.model_hashes[path] = gen_sha256(path)
        
        return self.model_hashes[path]

    def load_model(self, path: str, base: StableDiffusionBaseVersion) -> DiffusionPipeline:
        hash = gen_sha256(path)
        self.model_hashes[path] = hash

        if not hash in self.pipelines:
            print(f"Model '{path}' not cached! Loading model...")

            if base == StableDiffusionBaseVersion.SD1_5 or base == StableDiffusionBaseVersion.SD2_1:
                pipeline = StableDiffusionPipeline.from_single_file(path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
                pipeline.to("cuda")
                
                pipeline.enable_xformers_memory_efficient_attention()
                pipeline.enable_model_cpu_offload()

                self.pipelines[hash] = pipeline
            
            if base == StableDiffusionBaseVersion.SDXL1_0 or base == StableDiffusionBaseVersion.SDXLTurbo or base == StableDiffusionBaseVersion.SDXLLightning:
                pipeline = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
                pipeline.to("cuda")

                pipeline.enable_xformers_memory_efficient_attention()
                pipeline.enable_model_cpu_offload()

                self.pipelines[hash] = pipeline
            
            print("Done!")

        return self.pipelines[hash]

    # Stores model pipelines along their hashes
    pipelines: dict[str, DiffusionPipeline]

    # Stores model file paths along their hashes
    model_hashes: dict[str, str]

    max_models_in_vram: int = 2
