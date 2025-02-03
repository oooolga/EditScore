##############################
### author : Ge Ya (Olga) Luo
##############################

import pytest
import os

def test_get_baseline_edit():
    from editscore.baseline_models import InstructPix2Pix
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

    prompt = "turn him into cyborg"
    edited_image = model.get_editted_image(prompt, image)