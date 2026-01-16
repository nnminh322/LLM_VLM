from typing import Any, Dict, List
from core.base_processor import BaseProcessor
from core.base_inputs import TextInput
from transformers import AutoTokenizer


class LLMProcessor(BaseProcessor[TextInput]):
    def __init__(self, model_name: str, config: Dict):
        super().__init__(model_name, config)

    def setup(self):
        model_name = self.config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def processes(self, inputs: TextInput):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is None. Did you call setup()?")
        messages = self.apply_chat_template(inputs.messages)
        encoding = self.tokenizer(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=inputs.max_length,
        )
        return dict(encoding)

    def decode(self, token_ids):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is None.")
        return self.tokenizer.decode(token_ids=token_ids, skip_special_tokens=True)