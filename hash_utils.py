import hashlib

def gen_sha256(filepath):
    sha256_hash = hashlib.sha256()

    with open(filepath, "rb") as file:
        # Read the file in chunks of 4K
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()
