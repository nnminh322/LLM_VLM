import torch
from core.base_model import BaseModel
from core.base_processor import BaseProcessor
from core.base_inputs import BaseInput
from typing import Dict, Any


def preprocess(processor: BaseProcessor, inputs: BaseInput):
    return processor.processes(inputs=inputs)


def forward_train(model: BaseModel, batch: Dict[str, torch.Tensor]):
    return model.model(**batch)


@torch.no_grad()
def forward_infer(
    model: BaseModel, batch: Dict[str, torch.Tensor], generate_kwargs: Dict[str, Any]
):
    if generate_kwargs is None:
        generate_kwargs = {}
    return model.model.generate(**batch, **generate_kwargs)


def decode(processor: BaseProcessor, outputs):
    return processor.decode(outputs)


class Engine:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def train_step(self, inputs):
        batch = preprocess(processor=self.processor, inputs=inputs)
        return forward_train(model=self.model, batch=batch)

    def infer(self, inputs, **generate_kwargs):
        batch = preprocess(processor=self.processor, inputs=inputs)
        output = forward_infer(
            model=self.model, batch=batch, generate_kwargs=generate_kwargs
        )
        return output
