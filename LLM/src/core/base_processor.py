from abc import ABC, abstractmethod
from core.base_inputs import BaseInput
from typing import List, Dict, TypeVar, Generic, Optional
from transformers import PreTrainedTokenizerBase, ImageProcessingMixin
import torch

I = TypeVar("I", bound=BaseInput)


class BaseProcessor(Generic[I], ABC):
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.config = config
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.image_processor: Optional[ImageProcessingMixin] = None

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
