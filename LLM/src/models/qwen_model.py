import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.base_model import BaseModel
from core.registry import Registry
from utils.load_config import load_llm_config


@Registry.register("models", "qwen")
class QwenModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def load(self, mode="train") -> None:
        model_name = self.config["model_name"]
        """
        'train': Load 4-bit/8-bit với cấu hình QLoRA.
        'deploy': Load float16/bfloat16 tối ưu cho inference.
        """

        if mode == "train":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        elif mode == "deploy":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
