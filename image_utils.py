import os
import aiohttp
import aiofiles

import uuid

from PIL import ImageDraw, Image


def calculate_matrix_size(num_images: int):
    rows = 1
    columns = num_images

    for i in range(2, int(num_images**0.5) + 1):
        if num_images % i == 0:
            rows = i
            columns = num_images // i

    return rows, columns


def merge_images(images):
    num_images = len(images)
    rows, cols = calculate_matrix_size(num_images)

    image_width, image_height = images[0].size
    output_width = cols * image_width
    output_height = rows * image_height

    output_image = Image.new("RGB", (output_width, output_height), "black")
    draw = ImageDraw.Draw(output_image)

    for i in range(min(num_images, rows * cols)):
        row = i // cols
        col = i % cols
        x_position = col * image_width
        y_position = row * image_height
        output_image.paste(images[i], (x_position, y_position))

    return output_image


async def download_image(url: str, path: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                # Get the filename from the content disposition header
                content_disposition = response.headers.get("Content-Disposition")
                content_type = response.headers.get("Content-Type")

                print(content_disposition)
                filename = content_disposition.split("filename=")[-1].strip('"') if content_disposition else f"{uuid.uuid4().hex}.{content_type.split('/')[1]}"

                # Ensure the directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)

                # Save the image to the specified path with the extracted filename
                file_path = os.path.join(path, filename)
                async with aiofiles.open(file_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        await f.write(chunk)

                print(f"Image downloaded and saved to {file_path}")
                return file_path
            else:
                print(f"Failed to download image. Status code: {response.status}")
