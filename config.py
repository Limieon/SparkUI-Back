# SparkUI Backend Configuration

class SparkUIConfig:
    class API:
        HOST = "0.0.0.0"
        PORT = 1910
    class Directories:
        class StableDiffusion:
            CHECKPOINT = "assets/sd/checkpoint"
            LORA = "assets/sd/lora"
            EMBEDDING = "assets/sd/embedding"
            CONTROL_NET = "assets/sd/control_net"
