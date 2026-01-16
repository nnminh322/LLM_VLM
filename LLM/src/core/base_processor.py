from abc import ABC, abstractmethod
from core.base_inputs import BaseInput
from typing import List, Dict, TypeVar, Generic, Optional, Any
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
    def setup(self) -> None:
        # set tokenizer and image processor (maybe from huggingface)
        pass

    @abstractmethod
    # def processes(self, inputs: I) -> Dict[str, torch.Tensor]:
    def processes(self, inputs: I) -> Dict[str, Any]:  # fuk you pylance
        pass

    @abstractmethod
    def decode(self, token_ids: Any) -> str:
        pass

    def apply_chat_template(
        self, messages: List[Dict[str, str]], is_inference: bool = True
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer chưa được setup! Hãy gọi .setup() trước.")

        # Ép kiểu kết quả trả về chắc chắn là str
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=is_inference
        )
        return str(result)
