##############################
### author : Ge Ya (Olga) Luo
##############################

import pytest
import os

def test_use_gpt4o():

    # Example usage
    from editscore.mllm_tools.openai import GPT4v, GPT4o
    from editscore.scoring.prompt_tools import PromptList, PromptFSSample, MetricPrompt
    prompt_list = PromptList(mllm_model=GPT4v)
    prompt_list.append_raw_prompt("text", "Change the color of the zebra.")
    prompt_list.append_raw_prompt("image", 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg')
    prompt_list.postproccess()

    from editscore.baseline_models import InstructPix2Pix
    import PIL, requests
    model = InstructPix2Pix()
    def download_image(url):
        image = PIL.Image.open(requests.get(url, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image
    image = download_image('https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg')

    edited_image = model.get_editted_image("Change the color of the zebra.", image)

    fs_sample = PromptFSSample(mllm_model=GPT4v,
                               input_image='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
                               edit_instruction="Change the color of the zebra.",
                               ideal_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
                               new_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg',
                               baseline_edit=edited_image,
                               suggested_score=2,
                               reasoning="The zebra's shape changed.",)
    print(fs_sample._inner_list)

    metric_prompt = MetricPrompt(mllm_model=GPT4v)
    metric_prompt.add_fs_example(input_image='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
                                 edit_instruction="Change the color of the zebra.",
                                 ideal_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
                                 new_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg',
                                 baseline_edit=edited_image,
                                 suggested_score=2,
                                 reasoning="The zebra's shape changed.")
    metric_prompt.finalize(input_image='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
                            edit_instruction="Change the color of the zebra.",
                            ideal_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
                            new_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg',
                            baseline_edit=edited_image)
    for item in metric_prompt._inner_list:
        print(item)
    
    mllm = GPT4o('/home/mila/x/xuolga/keys/olga_personal_metric.env')
    prompt = mllm.prepare_prompt(metric_prompt)
    res = mllm.get_parsed_output(prompt)
    print("result : \n", res)