from unsloth import FastLanguageModel
from core.base_model import BaseModel
from core.registry import Registry
import torch
from typing import Any, Dict, cast


@Registry.register("models", "unsloth_qwen")
class UnslothQwenModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    def load(self, mode="train"):
        model_name = self.config["model_name"]
        max_seq_length = self.config.get("max_seq_length", 2048)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_8bit=True,
            device_map="auto",
        )
        self.model = model
        self.tokenizer = tokenizer

    def get_target_modules(self):
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        return target_modules

    def predict(self, records: Dict[str, torch.Tensor], **generate_kwargs):
        if self.model is None:
            raise RuntimeError("Model chưa được nạp. Hãy gọi .load() trước.")
        FastLanguageModel.for_inference(self.model)
        kwargs = {
            # "max_new_tokens": self.config.get("max_new_tokens", 128),
            "use_cache": True,
            **generate_kwargs
        }
        m: Any = self.model
        outputs = m.generate(**records, **kwargs)
        return cast(torch.Tensor, outputs)