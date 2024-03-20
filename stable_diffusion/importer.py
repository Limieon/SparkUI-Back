import os
import requests
import json

from dataclasses import dataclass

from hash_utils import get_sha256

from db_utils import add_image_by_url

from prisma.models import StableDiffusionBase, StableDiffusionCheckpoint, StableDiffusionCheckpointGroup, Image

from . import StableDiffusionBaseModel


@dataclass
class SDImportConfig:
    civitai_key: str
    models_dir: str
    checkpoints: list[str] | str
    embeddings: list[str] | str
    loras: list[str] | str
    lycorsis: list[str] | str
    control_nets: list[str] | str
    vaes: list[str] | str


def civitai_sd_to_spark_sd(v: str) -> str:
    if v == "SD 1.5":
        return "SD1_5"
    if v == "SDXL 1.0":
        return "SDXL1_0"
    if v == "SDXL Lightning":
        return "SDXL1_0-Ligthning"
    if v == "SDXL Turbo":
        return "SDXL1_0-Turbo"

    raise ValueError(f"Found invalid CivitAI version {v}!")


def spark_sd_to_base_enum(v: str) -> StableDiffusionBaseModel:
    if v == "SD1_5":
        return StableDiffusionBaseModel.SD1_5
    if v == "SDXL1_0":
        return StableDiffusionBaseModel.SDXL1_0
    if v == "SDXL1_0-Ligthning":
        return StableDiffusionBaseModel.SDXL1_0Lightning
    if v == "SDXL1_0-Turbo":
        return StableDiffusionBaseModel.SDXL1_0Turbo

    raise ValueError(f"Found invalid Spark SD Base version {v}!")


async def sd_import_models(config: SDImportConfig):
    if config.civitai_key is None:
        print("Cannot import models without valid CivitAI key!")
        print("Create one at https://civitai.com/user/account")
        return

    # Used to import my current test models, might not be available on your setup
    await import_model("./assets/models/StableDiffusion/dreamshaper_8.safetensors")
    await import_model("./assets/models/StableDiffusion/dreamshaperXL_v21TurboDPMSDE.safetensors")
    return

    for checkpoint_folder in config.checkpoints:
        checkpoints_path = os.path.join(config.models_dir, checkpoint_folder)

        for f in os.listdir(checkpoints_path):
            file = os.path.join(checkpoints_path, f)
            if os.path.isfile(file):
                if not file.endswith(".safetensors"):
                    continue

                await import_model(file)


async def import_model(file: str):
    file_hash = get_sha256(file)
    checkpoints = await StableDiffusionBase.prisma().find_first(where={"sha256": file_hash})

    if not checkpoints is None:
        print("Checkpoint already in database, skipping import!")
        return

    version_data = requests.get(f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}").json()
    model_data = requests.get(f"https://civitai.com/api/v1/models/{version_data['modelId']}").json()

    spark_version_id = (
        await StableDiffusionCheckpoint.prisma().create(
            {
                "baseID": (
                    await StableDiffusionBase.prisma().create(
                        {
                            "name": f"{model_data['name']} - {version_data['name']}",
                            "description": model_data["description"],
                            "file": file,
                            "format": "safetensors",
                            "sha256": file_hash,
                            "sdBaseModel": civitai_sd_to_spark_sd(version_data["baseModel"]),
                            "civitaiID": version_data["id"],
                            "originPage": f"https://civitai.com/models/f{model_data['id']}",
                            "civitaiData": json.dumps(model_data),
                        }
                    )
                ).id
            }
        )
    ).baseID

    for img in version_data["images"]:
        (
            url_split,
            width,
            height,
        ) = (
            img["url"].split("/"),
            img["width"],
            img["height"],
        )

        url = []

        for s in url_split:
            if not "width=" in s:
                url.append(s)

        image = await add_image_by_url(
            "/".join(url), os.path.join(os.getenv("SPARK_DIRS_IMAGES"), model_data["name"].replace(" ", "_"), version_data["name"]).replace(" ", "_")
        )
        await Image.prisma().update(where={"id": image.id}, data={"stableDiffusionBaseId": spark_version_id})
