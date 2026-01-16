from core.base_tuner import BaseTuner
from core.base_model import BaseModel
from core.registry import Registry
from unsloth import FastLanguageModel


@Registry.register("tunners", "qlora_unsloth")
class UnslothQLoRATunner(BaseTuner):
    def apply(self, wrapper_model: BaseModel, target_modules=None):
        assert wrapper_model.model is not None, "Model must be loaded before calling"
        target_modules = target_modules or wrapper_model.get_target_modules()
        model = wrapper_model.model

        model = FastLanguageModel.get_peft_model(
            model=model,
            r=self.config.get("lora_r", 8),
            target_modules=target_modules,
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0),
            bias=self.config.get("bias", "none"),
        )
        
        return model
