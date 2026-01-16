from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from core.base_trainer import BaseTrainer
import torch


class UnslothSFTTrainer(BaseTrainer):
    def train(self):
        config = SFTConfig(
            output_dir=self.config.get("output_dir", "outputs"),
            per_device_eval_batch_size=self.config.get("batch_size", 2),
            gradient_accumulation_steps=self.config.get(
                "gradient_accumulation_steps", 4
            ),
            learning_rate=self.config.get("learning_rate", 2e-4),
            max_steps=self.config.get("max_steps", 60),
            logging_steps=self.config.get("logging_steps", 1),
            
            
            dataset_text_field=self.config.get("dataset_text_field", "text"),
            max_length=self.config.get("max_length", 2048),
            packing=self.config.get("packing", False),
            
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        )
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=config,
            # formatting_func=self.config.get("formatting_func"),
        )
        
        trainer.train()
