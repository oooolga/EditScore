from editscore.scoring.metric.utils import (
    mllm_output_to_dict, download_image
)
import math, PIL, validators
from editscore.scoring.prompt_tools import PromptList, PromptFSSample, MetricPrompt, START_DELIMITER, END_DELIMITER

class EditScore:
    def __init__(self, backbone="gpt4o", baseline="ip2p", few_shot_csv="./assets/fs_prompt.csv",
                       key_path=None, delimiter=False) -> None:
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
        self.pc_prompt = MetricPrompt(mllm_model=self.mllm_model, delimiter=delimiter)
        self.preload_fs_from_csv(few_shot_csv)

    
    def reset_fs_examples(self):
        self.fs_input_images = []
        self.fs_edit_instructions = []
        self.fs_ideal_edits = []
        self.fs_new_edits = []
        self.fs_baseline_edits = []
        self.fs_suggested_scores = []
        self.fs_reasonings = []
    
    def open_local_image(self, image_path):
        if not isinstance(image_path, PIL.Image.Image) and not validators.url(image_path):
            return PIL.Image.open(image_path)
        return image_path

    def process_baseline_edits(self, fs_input_images, fs_edit_instructions, fs_baseline_edits):
        assert len(fs_baseline_edits) == 0 or len(fs_baseline_edits) == len(fs_input_images)
        if len(fs_baseline_edits) == 0:
            fs_baseline_edits = [None] * len(fs_input_images)
        
        for idx, (fs_input_image, fs_edit_instruction) in enumerate(zip(fs_input_images, fs_edit_instructions)):
            if isinstance(fs_baseline_edits[idx], str) or isinstance(fs_baseline_edits[idx], PIL.Image.Image):
                fs_baseline_edits[idx] = self.open_local_image(fs_baseline_edits[idx])
            else:
                fs_input_image = download_image(fs_input_image) if isinstance(fs_input_image, str) else fs_input_image
                fs_baseline_edits[idx] = self.baseline_model.get_editted_image(fs_edit_instruction, fs_input_image)
        return fs_baseline_edits
    
    def preload_fs_examples(self, fs_input_images,
                                  fs_edit_instructions,
                                  fs_ideal_edits,
                                  fs_new_edits,
                                  fs_baseline_edits,
                                  fs_suggested_scores,
                                  fs_reasonings):
        self.fs_input_images = [self.open_local_image(image_path) for image_path in fs_input_images]
        assert len(fs_input_images) == len(fs_edit_instructions) == len(fs_ideal_edits) == len(fs_new_edits) == len(fs_suggested_scores) == len(fs_reasonings)
        self.fs_edit_instructions = fs_edit_instructions
        self.fs_ideal_edits = [self.open_local_image(image_path) for image_path in fs_ideal_edits]
        self.fs_new_edits = [self.open_local_image(image_path) for image_path in fs_new_edits]
        self.fs_suggested_scores = fs_suggested_scores
        self.fs_reasonings = fs_reasonings
        self.fs_baseline_edits = self.process_baseline_edits(self.fs_input_images, self.fs_edit_instructions, fs_baseline_edits)
    
    def preload_fs_from_csv(self, csv_path="./assets/fs_prompt.csv"):
        self.reset_fs_examples()

        if  csv_path is None:
            return 

        import pandas as pd
        df = pd.read_csv(csv_path)
        self.preload_fs_examples(df['orig_image'].tolist(),
                                 df['edit_instruction'].tolist(),
                                 df['gt_edit'].tolist(),
                                 df['our_edit'].tolist(),
                                 df['baseline_edit'].tolist(),
                                 df['suggested_score'].tolist(),
                                 df['reasoning'].tolist())
        
    
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
                 fs_baseline_edits=[],
                 extract_overall_score_only=False, extract_all_score=True, echo_output=False):
        
        fs_baseline_edits = self.process_baseline_edits(fs_input_images, fs_edit_instructions, fs_baseline_edits)

        self.pc_prompt.add_fs_examples(input_images=fs_input_images+self.fs_input_images,
                                       edit_instructions=fs_edit_instructions+self.fs_edit_instructions,
                                       ideal_edits=fs_ideal_edits+self.fs_ideal_edits,
                                       new_edits=fs_new_edits+self.fs_new_edits,
                                       baseline_edits=fs_baseline_edits+self.fs_baseline_edits,
                                       suggested_scores=fs_suggested_scores+self.fs_suggested_scores,
                                       reasonings=fs_reasonings+self.fs_reasonings)
        
        input_image = self.open_local_image(input_image)
        edit_image = download_image(input_image) if isinstance(input_image, str) else input_image
        baseline_edit = self.baseline_model.get_editted_image(edit_instruction, edit_image)
        self.pc_prompt.finalize(input_image=input_image,
                                edit_instruction=edit_instruction,
                                ideal_edit=self.open_local_image(ideal_edit),
                                new_edit=self.open_local_image(new_edit),
                                baseline_edit=baseline_edit)
        prompt = self.mllm_model.prepare_prompt(self.pc_prompt)
        result = self.mllm_model.get_parsed_output(prompt) 
        results_dict = mllm_output_to_dict(result,
                                           start_delimiter=START_DELIMITER if self.pc_prompt.delimiter else None,
                                           end_delimiter=END_DELIMITER if self.pc_prompt.delimiter else None)

        self.pc_prompt.reinitialize()

    
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
    model = EditScore(key_path="/home/mila/x/xuolga/keys/vismin.env")
    import pdb; pdb.set_trace()
    eval = model.evaluate(
        input_image='./assets/org_28.png',
        edit_instruction="Can you make the bird white with black spots?",
        ideal_edit='./assets/gt_edited_28.png',
        new_edit='./assets/result_28_1.png',
    )
    print(eval)

    eval = model.evaluate(
        input_image='./assets/org_16.png',
        edit_instruction="Change the man into a woman.",
        ideal_edit='./assets/gt_edited_16.png',
        new_edit='./assets/result_16_1.png',
    )
    print(eval)
    import pdb; pdb.set_trace()