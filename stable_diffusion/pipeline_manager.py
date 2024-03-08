import torch

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline

from . import StableDiffusionBaseModel

from hash_utils import get_sha256


class PipelineManager:
    def get_model_hash(self, path: str) -> str:
        if not path in self.model_hashes:
            self.model_hashes[path] = get_sha256(path)

        return self.model_hashes[path]

    def load_pipeline(self, path: str, base: StableDiffusionBaseModel, use_gpu: bool) -> DiffusionPipeline:
        hash = self.get_model_hash(path)

        if not hash in self.pipelines:
            print(f"Model '{path}' not cached! Loading...")

            pipeline: DiffusionPipeline = None
            if base == StableDiffusionBaseModel.SD1_5 or base == StableDiffusionBaseModel.SD2_1:
                pipeline = StableDiffusionPipeline.from_single_file(path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

            if base == StableDiffusionBaseModel.SDXL1_0 or base == StableDiffusionBaseModel.SDXL1_0Turbo or base == StableDiffusionBaseModel.SDXL1_0Lightning:
                pipeline = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

            if use_gpu:
                pipeline.to("cuda")
                pipeline.enable_model_cpu_offload()
                pipeline.enable_xformers_memory_efficient_attention()

            self.pipelines[hash] = pipeline

            print("Done!")

        return self.pipelines[hash]

    model_hashes: dict[str, str] = {}
    pipelines: dict[str, DiffusionPipeline] = {}
