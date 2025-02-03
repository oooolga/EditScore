import PIL

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class InstructPix2Pix:
    
    def __init__(self, device="cuda"):
        self.model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
    
    def get_editted_image(self, prompt, image, num_inference_steps=10, image_guidance_scale=1):
        images = self.pipe(prompt, image=image, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale).images
        return images[0]

if __name__ == "__main__":
    import requests
    url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    def download_image(url):
        image = PIL.Image.open(requests.get(url, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image
    image = download_image(url)

    model = InstructPix2Pix()
    prompt = "turn him into cyborg"
    image = model.get_editted_image(prompt, image=image, num_inference_steps=10, image_guidance_scale=1)