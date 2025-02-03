from abc import ABC, abstractmethod
from typing import List, Any, Union
from editscore.scoring.prompt_tools import PromptList, PromptItem

class MLLM(ABC):
    def prepare_prompt(self, prompt_list: Union[PromptList, List[Any]]):
        ret = []
        if isinstance(prompt_list, PromptList):
            prompt_list = prompt_list._inner_list
        
        for prompt in prompt_list:
            if isinstance(prompt, PromptList) or isinstance(prompt, list):
                ret += self.prepare_prompt(prompt)
            else:
                assert isinstance(prompt, PromptItem)
                if not prompt.postprocessed:
                    prompt.postprocess(self)
                ret.append(prompt.content)
        return ret
    
    def prepare_text_prompt(prompt):
        return prompt

    @staticmethod
    def prepare_image_prompt(prompt):
        raise NotImplementedError("Subclasses should implement this!")