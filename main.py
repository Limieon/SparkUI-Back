from stable_diffusion import StableDiffusionQueue, StableDiffusionBaseVersion

sd_queue = StableDiffusionQueue()

# Prompt Credit: https://civitai.com/images/7260553
prompt = "anime, girl, wizard hat, robe, thighhighs, close-up, happy, magic, fire, bokeh, depth of field, transparent, light particles, bloom effect"
sd_queue.load_model(path="./assets/models/StableDiffusion/bluePencilXL_v500.safetensors", base=StableDiffusionBaseVersion.SDXL1_0)(prompt=prompt, guidance_scale=5, num_inference_steps=20).images[0].save("test.png")
sd_queue.load_model(path="./assets/models/StableDiffusion/bluePencilXL_v500.safetensors", base=StableDiffusionBaseVersion.SDXL1_0)(prompt=prompt, width=576, height=1024, guidance_scale=5, num_inference_steps=20).images[0].save("test2.png")
