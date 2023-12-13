import os

import db
import api

from dotenv import load_dotenv

load_dotenv()

SD_ASSET_DIR = os.getenv("SPARKUI_BACK_SD_DIR")

SD_CHECKPOINT_DIR = os.path.join(SD_ASSET_DIR, "checkpoints")
SD_LORA_DIR = os.path.join(SD_ASSET_DIR, "lora")
SD_EMBEDDING_DIR = os.path.join(SD_ASSET_DIR, "embedding")
SD_VAE_DIR = os.path.join(SD_ASSET_DIR, "vae")

def main():
    for dir in [ SD_CHECKPOINT_DIR,SD_LORA_DIR,SD_EMBEDDING_DIR, SD_VAE_DIR ]:
        if not os.path.exists(dir): os.makedirs(dir)
    
    print("Starting web server...")
    api.init()

if __name__ == "__main__":
    main()
