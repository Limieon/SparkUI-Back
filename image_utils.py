from PIL import ImageDraw, Image


def merge_images(images: list[Image.Image]):
    num_images = len(images)

    if num_images <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 3, 3

    # Calculate the size of the output image
    image_width, image_height = images[0].size
    output_width = cols * image_width
    output_height = rows * image_height

    # Create a blank white image as the base for merging
    output_image = Image.new("RGB", (output_width, output_height), "white")
    draw = ImageDraw.Draw(output_image)

    # Paste each image onto the output image in a matrix form
    for i in range(min(num_images, rows * cols)):
        row = i // cols
        col = i % cols
        x_position = col * image_width
        y_position = row * image_height
        output_image.paste(images[i], (x_position, y_position))

    return output_image
