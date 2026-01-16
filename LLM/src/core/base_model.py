from abc import ABC, abstractmethod
from typing import Literal,Optional
import torch.nn as nn

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.model: Optional[nn.Module] = None
        # self.tokenizer = None

    @abstractmethod
    def load(self, mode: Literal["train", "deploy"] = "train") -> None:
        # Load model from config
        pass

    @abstractmethod
    def get_target_module(self):  # Get layer module for apply LoRA/QLoRa
        pass

    # @abstractmethod
    # def predict(self, records):
    #     # Using for inference
    #     pass
