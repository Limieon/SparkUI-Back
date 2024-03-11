import math
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
