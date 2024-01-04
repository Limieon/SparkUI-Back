# SparkUI Backend Configuration
class SD_BaseVersion:
    def __init__(self, handle: str, name: str, short: str):
        self.handle = handle
        self.name = name
        self.short = short

    handle: str
    name: str
    short: str


class SD_Sampler:
    def __init__(self, handle: str, group: str, name: str):
        self.handle = handle
        self.group = group
        self.name = name

    handle: str
    group: str
    name: str


class SparkUIConfig:
    class Frontend:
        HOST = "http://127.0.0.1"
        PORT = 1910

    class API:
        HOST = "0.0.0.0"  # The host to bind the API server on
        PORT = 1911  # The port to expose the API server to

    class StableDiffusion:
        class Directories:
            CHECKPOINT = "assets/sd/checkpoint"  # The checkpoint asset dir
            LORA = "assets/sd/lora"  # The lora asset dir
            EMBEDDING = "assets/sd/embedding"  # The embedding asset dir
            CONTROL_NET = "assets/sd/control_net"  # The controlnet root dir
            VAE = "assets/sd/vae"  # The vae asset dir
            IMAGES_OUT = "assets/out/images"  # The image out dir

        class BaseModels:
            SD1_5 = SD_BaseVersion("sd1_5", "StableDiffusion 1.5", "SD1.5")
            SD2_1 = SD_BaseVersion("sd2_1", "StableDiffusion 2.1", "SD2.1")
            SDXL1 = SD_BaseVersion("sdxl", "StableDiffusionXL", "SDXL")
            SDXLT = SD_BaseVersion("sdxlt", "StableDiffusionXL - Turbo", "SDXL-T")

        class Samplers:  # More samplers will be added in the future and the samplers will be stored somewhere else
            EULER_A = SD_Sampler("euler_a", "Euler", "Euler Ancestral")
            DPMPP_2M = SD_Sampler("dpmpp_2m", "DPM++", "DPM++ 2M")
            DPMPP_2M_KARRAS = SD_Sampler("dpmpp_2m_karras", "DPM++", "DPM++ 2M Karras")

        MAX_LOADED_CHECKPOINTS = 1
