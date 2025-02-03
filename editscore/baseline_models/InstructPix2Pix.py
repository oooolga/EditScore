from editscore.baseline_models.baseline_abstract import ModelBase

class InstructPix2Pix(ModelBase):
    def __init__(self, save_dir=None, num_inference_steps=10, image_guidance_scale=1):
        from editscore.ip2p import InstructPix2Pix
        ip2p = InstructPix2Pix()
        super().__init__(model=ip2p, save_dir=save_dir)
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
    
    def get_editted_image(self, prompt, original_image, filename=None):
        ret = self.model.get_editted_image(prompt,
                                           image=original_image,
                                           num_inference_steps=self.num_inference_steps,
                                           image_guidance_scale=self.image_guidance_scale)
        if filename and self.save_dir:
            ret.save(f"{self.save_dir}/{filename}.jpg")
        return ret

if __name__ == "__main__":
    model = InstructPix2Pix()
    import PIL
    import requests

    url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    def download_image(url):
        image = PIL.Image.open(requests.get(url, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image
    image = download_image(url)

    edited_image = model.get_editted_image(prompt, image)