from typing import Any, Dict, List
from core.base_processor import BaseProcessor
from core.base_inputs import TextInput
from transformers import AutoTokenizer


class LLMProcessor(BaseProcessor):
    def __init__(self, model_name: str, config: Dict):
        super().__init__(model_name, config)

    def setup(self):
        model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def processes(self, inputs: TextInput):
        return self.tokenizer(
            inputs.messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=inputs.max_length,
        )

    def decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
