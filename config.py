# SparkUI Backend Configuration

class SparkUIConfig:
    class API:
        HOST = "0.0.0.0"                                # The host to bind the API server on
        PORT = 1910                                     # The port to expose the API server to
    class Directories:
        class StableDiffusion:
            CHECKPOINT = "assets/sd/checkpoint"         # The checkpoint asset dir
            LORA = "assets/sd/lora"                     # The lora asset dir 
            EMBEDDING = "assets/sd/embedding"           # The embedding asset dir
            CONTROL_NET = "assets/sd/control_net"       # The controlnet root dir
            VAE = "assets/sd/vae"                       # The vae asset dir
            IMAGES_OUT = "assets/out/images"            # The image out dir
