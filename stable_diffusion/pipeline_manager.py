import os
import asyncio
import torch

from fastapi import HTTPException
from queue import Queue
from typing import Callable
from PIL.Image import Image
from DeepCache import DeepCacheSDHelper

from compel import Compel, ReturnedEmbeddingsType

import prisma.models as pm

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from .generation_request import Txt2ImgRequest
from . import StableDiffusionBaseModel
from .importer import spark_sd_to_base_enum

from hash_utils import get_sha256


def get_pipeline_embeds(pipeline: DiffusionPipeline, prompt: str, negative_prompt: str, device: str):
    """Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    count_prompt = len(prompt.split(" "))
    count_negative_prompt = len(negative_prompt.split(" "))

    # create the tensor based on which prompt is longer
    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(
            negative_prompt, truncation=False, padding="max_length", max_length=shape_max_length, return_tensors="pt"
        ).input_ids.to(device)

    else:
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length", max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i : i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i : i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)


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
                pipeline = StableDiffusionPipeline.from_single_file(
                    path, torch_dtype=torch.float16 if use_gpu else torch.float32, use_safetensors=True, safety_checker=None
                )
                pipeline.safety_checker = None

            if base == StableDiffusionBaseModel.SDXL1_0 or base == StableDiffusionBaseModel.SDXL1_0Turbo or base == StableDiffusionBaseModel.SDXL1_0Lightning:
                pipeline = StableDiffusionXLPipeline.from_single_file(
                    path,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    use_safetensors=True,
                    use_karras_sigmas=True,
                    euler_at_final=True,
                    scheduler=DPMSolverMultistepScheduler(solver_order=2),
                    safety_checker=None,
                )
                pipeline.safety_checker = None

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

            try:
                ckpt = await pm.StableDiffusionCheckpoint.prisma().find_first_or_raise(
                    where={
                        "baseID": gen_data.checkpoint_id,
                    },
                    include={"base": {}},
                )
            except Exception as e:
                print(f"Error finding checkpoint: {e}")

            pipe = self.pipeline_manager.load_pipeline(
                ckpt.base.file, spark_sd_to_base_enum(ckpt.base.sdBaseModel), use_gpu=True if os.getenv("SPARK_USE_GPU") == "true" else False
            )

            for lora in gen_data.loras:
                pipe.load_lora_weights("./assets/models/Lora", weight_name=lora.lora, weight=lora.weight)

            if isinstance(pipe, StableDiffusionXLPipeline):
                print("Detected SDXL, using 2 tokenizers and encoders...")
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=True,
                    device="cuda",
                )

                conditioning, pooled = compel(gen_data.prompt)
                neg_conditioning, neg_pooled = compel("" if gen_data.negative_prompt is None else gen_data.negative_prompt)
                [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

                images = pipe(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=neg_conditioning,
                    negative_pooled_prompt_embeds=neg_pooled,
                    num_inference_steps=gen_data.steps,
                    guidance_scale=gen_data.cfg_scale,
                    width=gen_data.width,
                    height=gen_data.height,
                    num_images_per_prompt=gen_data.num_images,
                ).images
            elif isinstance(pipe, StableDiffusionPipeline):
                print("Detected SD, using single tokenizer and encoder...")
                compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, device="cuda", truncate_long_prompts=True)
                conditioning = compel.build_conditioning_tensor(gen_data.prompt)
                neg_conditioning = compel.build_conditioning_tensor("" if gen_data.negative_prompt is None else gen_data.negative_prompt)
                [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

                images = pipe(
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=neg_conditioning,
                    num_inference_steps=gen_data.steps,
                    guidance_scale=gen_data.cfg_scale,
                    width=gen_data.width,
                    height=gen_data.height,
                    num_images_per_prompt=gen_data.num_images,
                ).images

            self.results[gen_data] = images

            result_ready.set()

    queue: Queue[tuple[Txt2ImgRequest, asyncio.Event]]
    pipeline_manager: PipelineManager
    results: dict[Txt2ImgRequest, list[Image]]
