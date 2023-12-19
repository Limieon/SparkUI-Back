# SparkUI Backend Configuration

class SD_BaseVersion:
    def __init__(self, handle: str, name: str, short: str):
        self.handle = handle
        self.name = name
        self.short = short

    handle: str
    name: str
    short: str

class SparkUIConfig:
    class API:
        HOST = "0.0.0.0"                                # The host to bind the API server on
        PORT = 1910                                     # The port to expose the API server to

    class StableDiffusion:
        class Directories:
            CHECKPOINT = "assets/sd/checkpoint"         # The checkpoint asset dir
            LORA = "assets/sd/lora"                     # The lora asset dir 
            EMBEDDING = "assets/sd/embedding"           # The embedding asset dir
            CONTROL_NET = "assets/sd/control_net"       # The controlnet root dir
            VAE = "assets/sd/vae"                       # The vae asset dir
            IMAGES_OUT = "assets/out/images"            # The image out dir
        
        class BaseModels:
            SD1_5: SD_BaseVersion("sd1_5", "StableDiffusion 1.5", "SD1.5")
            SD2_1: SD_BaseVersion("sd2_1", "StableDiffusion 2.1", "SD2.1")
            SDXL1: SD_BaseVersion("sdxl", "StableDiffusionXL", "SDXL")
            SDXLT: SD_BaseVersion("sdxlt", "StableDiffusionXL - Turbo", "SDXL-T")
