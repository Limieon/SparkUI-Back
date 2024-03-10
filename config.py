class SparkConfig:
    class API:
        host = "0.0.0.0"
        port = 1911

    class Database:
        host = "127.0.0.1"
        port = 5432

    class Directories:
        models = "./assets/models"
        outputs = "./assets/outputs"

        class StableDiffusionFolders:
            embedding = ["TextualInversion"]
            checkpoint = ["StableDiffusion"]
            control_net = ["Controlnet"]
            lora = ["Lora"]
            lycorsis = ["LyCORSIS"]
            vae = ["VAE"]
