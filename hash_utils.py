import hashlib
import base64

from PIL.Image import Image


def get_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()

    with open(filepath, "rb") as file:
        # Read the file in chunks of 4K
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def encode_image_to_base64(image: Image):
    try:
        encoded_image = base64.b64encode(image.tobytes())
        encoded_string = encoded_image.decode("utf-8")
        return encoded_string

    except Exception as e:
        print(f"Error encoding image: {e}")
        return None
