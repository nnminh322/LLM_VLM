from abc import ABC, abstractmethod
from typing import Literal,Optional, List, Dict, Any
import torch.nn as nn
import torch

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
    def get_target_modules(self) -> List[str]:  # Get layer module for apply LoRA/QLoRa
        pass
    
    @abstractmethod
    def predict(self, records: Dict[str, torch.Tensor], **kwargs: Any) -> Any:
        pass
