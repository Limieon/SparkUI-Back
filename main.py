import os

import api

from config import SparkUIConfig as Config

def main() -> None:
    print("Initializing Directories...")
    for field, value in vars(Config.Directories.StableDiffusion).items():
        if field.startswith("__"): continue
        path = os.path.join(".", value)
        if not os.path.exists(path):
            os.makedirs(path)

    print("Starting API...")
    api.init()

if __name__ == '__main__':
    main()
