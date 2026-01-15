from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self):
        # Load model from config
        pass

    @abstractmethod
    def get_target_module(self):  # Get layer module for apply LoRA/QLoRa
        pass

    @abstractmethod
    def predict(self, records):
        # Using for inference
        pass
