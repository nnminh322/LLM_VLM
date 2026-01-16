from processors.llm_processor import LLMProcessor
from utils.load_config import load_llm_config
from core.registry import Registry

def build_model(name: str):
    config = load_llm_config(name)
    model_cls = Registry.get("models", name=name)
    if model_cls is None:
        raise ValueError(f"Model '{name}' not registered!")
    model = model_cls(config)
    model.load()
    return model


def build_processor(name: str):
    config = load_llm_config(name)
    return LLMProcessor(model_name=name, config=config)

def build_model_and_processor(name: str):
    config = load_llm_config(name)

    model_cls = Registry.get("models", name=name)
    if model_cls is None:
        raise ValueError(f"Model '{name}' not registered!")

    model = model_cls(config)
    model.load()

    processor = LLMProcessor(model_name=name, config=config)
    processor.setup()

    return model, processor

