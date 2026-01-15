from typing import Any, Dict, List
from LLM.src.core.base_inputs import TextInput
from torch._tensor import Tensor
from core.base_processor import BaseProcessor
from transformers import AutoTokenizer


class LLMProcessor(BaseProcessor):
    def __init__(self, model_id: str, config: Dict):
        super().__init__(model_id, config)

    def setup(self):
        model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )


