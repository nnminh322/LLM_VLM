from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch


class BaseProcessor(ABC):
    def __init__(self, model_id: str, config: Dict):
        self.model_id = model_id
        self.config = config
        self.tokenizer = None
        self.image_processor = None

    @abstractmethod
    def setup(self):
        # set tokenizer and image processor (maybe from huggingface
        pass

    @abstractmethod
    def processes(
        self,
        messages: List[Dict[str, str]],
        image: Optional[List[Any]] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        pass 
    
    @abstractmethod
    def decode(self, token_ids):
        pass
    
    def apply_chat_tempplate(self, messages: List[Dict[str, str]]):
        # will complete later
        pass
