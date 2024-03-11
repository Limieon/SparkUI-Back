from dataclasses import dataclass


@dataclass
class SDImportConfig:
    models_dir: str
    checkpoints: list[str]
    embeddings: list[str]
    loras: list[str]
    lycorsis: list[str]
    control_nets: list[str]
    vaes: list[str]


async def sd_import_models(config: SDImportConfig):
    print(config)
