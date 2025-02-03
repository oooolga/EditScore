from abc import ABC, abstractmethod

class ModelBase(ABC):
    def __init__(self, model, save_dir=None):
        self.model = model
        self.save_dir = save_dir
    
    def get_editted_image(self, prompt, original_image, filename=None):
        raise NotImplementedError("Subclasses should implement this!")