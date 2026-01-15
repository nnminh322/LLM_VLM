from abc import ABC, abstractmethod
from LLM.src.core.base_inputs import BaseInput
from typing import List, Dict, Union, TypeVar, Generic
import torch

I = TypeVar("I", bound=BaseInput)


class BaseProcessor(Generic[I], ABC):
    def __init__(self, model_id: str, config: Dict):
        self.model_id = model_id
        self.config = config
        self.tokenizer = None
        self.image_processor = None

    @abstractmethod
    def setup(self):
        # set tokenizer and image processor (maybe from huggingface)
        pass

    @abstractmethod
    def processes(self, inputs: I) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def decode(self, token_ids):
        pass

    def apply_chat_tempplate(self, messages: List[Dict[str, str]]):
        # will complete later
        pass
