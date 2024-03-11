import asyncio
import torch

from queue import Queue
from typing import Callable
from PIL.Image import Image
from DeepCache import DeepCacheSDHelper

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline

from .generation_request import Txt2ImgRequest
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
                pipeline = StableDiffusionPipeline.from_single_file(path, torch_dtype=torch.float16 if use_gpu else torch.float32, use_safetensors=True)

            if base == StableDiffusionBaseModel.SDXL1_0 or base == StableDiffusionBaseModel.SDXL1_0Turbo or base == StableDiffusionBaseModel.SDXL1_0Lightning:
                pipeline = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=torch.float16 if use_gpu else torch.float32, use_safetensors=True)

            
            if use_gpu:
                pipeline.to("cuda")
                pipeline.enable_model_cpu_offload()
                pipeline.enable_xformers_memory_efficient_attention()

            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(
                cache_interval=3,
                cache_branch_id=0,
            )
            helper.enable()

            self.pipelines[hash] = pipeline

            print("Done!")

        return self.pipelines[hash]

    model_hashes: dict[str, str] = {}
    pipelines: dict[str, DiffusionPipeline] = {}


class GenerationQueue:
    def __init__(self, pipeline_manager: PipelineManager):
        self.pipeline_manager = pipeline_manager
        self.queue = Queue()
        self.results = {}

    async def queue_txt2img(self, generation_request: Txt2ImgRequest):
        result_ready = asyncio.Event()
        self.queue.put((generation_request, result_ready))
        await result_ready.wait()

        images = self.results.pop(generation_request, None)
        return images

    def start_queue(self):
        asyncio.ensure_future(self.process_queue())

    async def process_queue(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(1)
                continue

            gen_data, result_ready = self.queue.get()

            pipe = self.pipeline_manager.load_pipeline(gen_data.checkpoint, StableDiffusionBaseModel.SD1_5, False)

            for lora in gen_data.loras:
                pipe.load_lora_weights("./assets/models/Lora", weight_name=lora.lora, weight=lora.weight)

            images = pipe(
                prompt=gen_data.prompt,
                negative_prompt=gen_data.negative_prompt,
                num_inference_steps=gen_data.steps,
                guidance_scale=gen_data.cfg_scale,
                width=gen_data.width,
                height=gen_data.height,
                num_images_per_promt=gen_data.num_images,
            ).images

            self.results[gen_data] = images

            result_ready.set()

    queue: Queue[tuple[Txt2ImgRequest, asyncio.Event]]
    pipeline_manager: PipelineManager
    results: dict[Txt2ImgRequest, list[Image]]
