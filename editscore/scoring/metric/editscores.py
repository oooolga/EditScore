from editscore.scoring.metric.utils import (
    mllm_output_to_dict, download_image
)
import math
from editscore.scoring.prompt_tools import PromptList, PromptFSSample, MetricPrompt, START_DELIMITER, END_DELIMITER

class EditScore:
    def __init__(self, backbone="gpt4o", baseline="ip2p", key_path=None, delimiter=False) -> None:
        self.backbone_name = backbone
        self.baseline_name = baseline

        if self.backbone_name == "gpt4o":
            from editscore.mllm_tools.openai import GPT4o
            self.mllm_model = GPT4o(key_path)
        elif self.backbone_name == "gpt4v":
            from editscore.mllm_tools.openai import GPT4v
            self.mllm_model = GPT4v(key_path)
        elif self.backbone_name == "gemini":
            from editscore.mllm_tools.openai import Gemini
            self.mllm_model = Gemini()
        # elif self.backbone_name == "idefics2":
        #     from mllm_tools.idefics2_eval import Idefics2
        #     self.model = Idefics2()
        # elif self.backbone_name == "mantis":
        #     from mllm_tools.mantis_idefics2_eval import Mantis
        #     self.model = Mantis()
        # elif self.backbone_name == "minicpmv":
        #     from mllm_tools.minicpmv_eval import MiniCPMV
        #     self.model = MiniCPMV()
        else:
            raise NotImplementedError("backbone not supported")

        if self.baseline_name == "ip2p":
            from editscore.baseline_models import InstructPix2Pix
            self.baseline_model = InstructPix2Pix()
        else:
            raise NotImplementedError("baseline not supported")

        from editscore.scoring.prompt_tools import MetricPrompt
        self.metric_prompt = MetricPrompt(mllm_model=self.mllm_model, delimiter=delimiter)
        

    def evaluate(self,
                 input_image,
                 edit_instruction,
                 ideal_edit,
                 new_edit,
                 fs_input_images=[],
                 fs_edit_instructions=[],
                 fs_ideal_edits=[],
                 fs_new_edits=[],
                 fs_suggested_scores=[],
                 fs_reasonings=[],
                 extract_overall_score_only=False, extract_all_score=True, echo_output=False):
        
        fs_baseline_edits = []
        for fs_input_image, fs_edit_instruction in zip(fs_input_images, fs_edit_instructions):
            fs_input_image = download_image(fs_input_image) if isinstance(fs_input_image, str) else fs_input_image
            fs_baseline_edits.append(self.baseline_model.get_editted_image(fs_edit_instruction, fs_input_image))
        
        self.metric_prompt.add_fs_examples(input_images=fs_input_images,
                                          edit_instructions=fs_edit_instructions,
                                          ideal_edits=fs_ideal_edits,
                                          new_edits=fs_new_edits,
                                          baseline_edits=fs_baseline_edits,
                                          suggested_scores=fs_suggested_scores,
                                          reasonings=fs_reasonings)
        
        edit_image = download_image(input_image) if isinstance(input_image, str) else input_image
        baseline_edit = self.baseline_model.get_editted_image(edit_instruction, edit_image)
        self.metric_prompt.finalize(input_image=input_image,
                                    edit_instruction=edit_instruction,
                                    ideal_edit=ideal_edit,
                                    new_edit=new_edit,
                                    baseline_edit=baseline_edit)
        prompt = self.mllm_model.prepare_prompt(self.metric_prompt)
        result = self.mllm_model.get_parsed_output(prompt) 
        results_dict = mllm_output_to_dict(result,
                                           start_delimiter=START_DELIMITER if self.metric_prompt.delimiter else None,
                                           end_delimiter=END_DELIMITER if self.metric_prompt.delimiter else None)

        self.metric_prompt.reinitialize()

    
        # SC_dict = False
        # PQ_dict = False
        # tries = 0
        # max_tries = 1
        # while SC_dict is False or PQ_dict is False:
        #     tries += 1
        #     guess_if_cannot_parse = True if tries > max_tries else False
        #     result_SC = self.model.get_parsed_output(SC_prompt_final)
        #     result_PQ = self.model.get_parsed_output(PQ_prompt_final)
        #     SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)
        #     PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=guess_if_cannot_parse)

        # if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
        #     print("rate_limit_exceeded") 
        #     raise ValueError("rate_limit_exceeded")
        # results_dict['SC'] = SC_dict
        # results_dict['PQ'] = PQ_dict
        # SC_score = min(results_dict['SC']['score'])
        # PQ_score = min(results_dict['PQ']['score'])
        # O_score = math.sqrt(SC_score * PQ_score)
        # results_dict['O'] = {'score': O_score}

        # if echo_output:
        #     print("results_dict", results_dict)
        # if extract_all_score:
        #     return [SC_score, PQ_score, O_score]
        # if extract_overall_score_only:
        #     return O_score
        return results_dict

if __name__ == "__main__":
    model = EditScore(key_path="/home/mila/x/xuolga/keys/olga_personal_metric.env")
    model.evaluate(
        input_image='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg',
        edit_instruction="Change the color of the zebra.",
        ideal_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg',
        new_edit='https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg',
        fs_input_images=['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg'],
        fs_edit_instructions=["Change the color of the zebra."],
        fs_ideal_edits=['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg'],
        fs_new_edits=['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_2.jpg'],
        fs_suggested_scores=[2],
        fs_reasonings=["The zebra's shape changed."],
    )

