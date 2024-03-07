from stable_diffusion import StableDiffusionHandler, StableDiffusionBaseModel


def main():
    sd_handler = StableDiffusionHandler()

    # Prompt Credit: https://civitai.com/images/7260553
    prompt = "anime, girl, wizard hat, robe, thighhighs, close-up, happy, magic, fire, bokeh, depth of field, transparent, light particles, bloom effect"
    sd_handler.load_pipeline("./assets/models/StableDiffusion/bluePencilXL_v500.safetensors", StableDiffusionBaseModel.SDXL1_0)(
        prompt, num_inference_steps=20, guidance_scale=5
    ).images[0].save("assets/outputs/image.png")

    sd_handler.load_pipeline("./assets/models/StableDiffusion/bluePencilXL_v500.safetensors", StableDiffusionBaseModel.SDXL1_0)(
        prompt, num_inference_steps=20, width=576, height=1024, guidance_scale=5
    ).images[0].save("assets/outputs/image2.png")


if __name__ == "__main__":
    main()
