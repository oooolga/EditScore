class PromptItem:
    def __init__(self, prompt_type="text", content=None, postprocessed=False):
        assert prompt_type in ["text", "image"]#, "list"]
        self.prompt_type = prompt_type
        self.content = content
        self.postprocessed = postprocessed
    
    def set_content(self, content):
        self.content = content
    
    def __str__(self):
        return f"PromptItem({self.prompt_type}, {self.content})"
    
    def postprocess(self, mllm_model):
        
        if not self.postprocessed:
            if self.prompt_type == "text":
                self.content = mllm_model.prepare_text_prompt(self.content)
            elif self.prompt_type == "image":
                self.content = mllm_model.prepare_image_prompt(self.content)
            # else:
            #     for idx in range(len(self.content)):
            #         assert isinstance(self.content[idx], PromptItem)
            #         self.content[idx] = self.content[idx].postprocess(mllm_model=mllm_model)
            self.postprocessed = True


class PromptList:
    def __init__(self, mllm_model):
        self._inner_list = list()
        self.mllm = mllm_model

    def __len__(self):
        return len(self._inner_list)

    def __delitem__(self, index):
        self._inner_list.__delitem__(index)

    def insert(self, index, value):
        assert isinstance(value, PromptItem)
        self._inner_list.insert(index, value)

    def __setitem__(self, index, value):
        assert isinstance(value, PromptItem)
        self._inner_list.__setitem__(index, value)

    def __getitem__(self, index):
        return self._inner_list.__getitem__(index)

    def append(self, value):
        assert isinstance(value, PromptItem)
        self.insert(len(self) + 1, value)
    
    def append_raw_prompt(self, prompt_type, content, postprocessed=False):
        assert prompt_type in ["text", "image"]
        value = PromptItem(
            prompt_type=prompt_type,
            content=content,
            postprocessed=postprocessed
        )
        self.append(value)
    
    def postproccess(self):
        for item in self._inner_list:
            item.postprocess(self.mllm)
    
    def __add__(self, other):
        assert isinstance(other, PromptList)
        new_list = PromptList(mllm_model=self.mllm)
        new_list._inner_list = self._inner_list + other._inner_list
        return new_list


class PromptFSSample(PromptList):
    def __init__(self, mllm_model,
                       input_image,
                       edit_instruction,
                       ideal_edit,
                       new_edit,
                       baseline_edit,
                       suggested_score=None,
                       reasoning=None,
                       highest_score=5,
                       baseline_score=3,
                       output_format="\"Score\" : {score}\n, \"Reasoning\" :  \"{reasoning}\""):
        super().__init__(mllm_model)
        self.fs_flag = not suggested_score is None
        if self.fs_flag:
            assert not output_format is None
            suggested_output = "{" + output_format.format(score=suggested_score, reasoning=reasoning) + "}"
        self.template = [('text', f"Edit prompt: {edit_instruction}\nOriginal image:"),
                         ('image', input_image),
                         ('text', f"Ideal edit image (score={highest_score}):"),
                         ('image', ideal_edit),
                         ('text', f"Baseline edit image (score={baseline_score}):"),
                         ('image', baseline_edit),
                         ('text', "Evalute the following image:"),
                         ('image', new_edit)]
        if self.fs_flag:
            self.template.append(('text', f"This example's output would be:\n{suggested_output}"))
        self.convert_template_to_prompt_list()

    
    def convert_template_to_prompt_list(self):
        for prompt_type, content in self.template:
            self.append_raw_prompt(prompt_type, content)
        self.postproccess()


class PromptStarter(PromptItem):

    def __init__(self):
        from .prompt_context import START_CONTEXT
        super().__init__(prompt_type="text",
                         content=START_CONTEXT)


class PromptRule(PromptItem):
    
    def __init__(self):
        from .prompt_context import PC_RULES_CONTEXT
        super().__init__(prompt_type="text",
                         content=PC_RULES_CONTEXT)

class MetricPrompt(PromptList):

    def __init__(self, mllm_model):
        super().__init__(mllm_model)
        prompt_starter = PromptStarter()
        prompt_rule = PromptRule()
        self.append(prompt_starter)
        self.append(prompt_rule)
        self.num_fs_samples = 0
        self.ready = False
        
    def add_fs_example(self, input_image, edit_instruction, ideal_edit, new_edit, baseline_edit, suggested_score, reasoning):
        from .prompt_context import FS_PREFIX_CONTENT
        assert not self.ready
        if self.num_fs_samples == 0:
            self.append(PromptItem(prompt_type="text", content=FS_PREFIX_CONTENT))
        
        self.append(PromptItem(prompt_type="text", content=f"Example {self.num_fs_samples + 1}:"))
        fs_sample = PromptFSSample(mllm_model=self.mllm,
                                   input_image=input_image,
                                   edit_instruction=edit_instruction,
                                   ideal_edit=ideal_edit,
                                   new_edit=new_edit,
                                   baseline_edit=baseline_edit,
                                   suggested_score=suggested_score,
                                   reasoning=reasoning)
        self._inner_list += fs_sample._inner_list
        self.num_fs_samples += 1
    
    def finalize(self, input_image, edit_instruction, ideal_edit, new_edit, baseline_edit):
        
        from .prompt_context import EVAL_PREFIX_CONTENT
        self.append(PromptItem(prompt_type="text", content=EVAL_PREFIX_CONTENT))
        fs_sample = PromptFSSample(mllm_model=self.mllm,
                                   input_image=input_image,
                                   edit_instruction=edit_instruction,
                                   ideal_edit=ideal_edit,
                                   new_edit=new_edit,
                                   baseline_edit=baseline_edit)
        self._inner_list += fs_sample._inner_list
        self.postproccess()
        self.ready = True
    
    def __add__(self, other):
        ret = super().__add__(other)
        ret.num_fs_samples = self.num_fs_samples
        ret.ready = False
        return ret


if __name__ == "__main__":

    # Example usage
    from editscore.mllm_tools.openai import GPT4v
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
