class Registry:
    _REGISTIES = {"models": {}, "datasets": {}, "tuners": {}, "tasks": {}}

    @classmethod
    def register(cls, catergory: str, name: str):
        def decorator(wrapper_class):
            if wrapper_class not in cls._REGISTIES:
                raise ValueError(f"wrapper_class: {wrapper_class} not in categories!")
            cls._REGISTIES[catergory][name] = wrapper_class
            return wrapper_class

        return decorator

    @classmethod
    def get(cls, category: str, name: str):
        return cls._REGISTIES[category].get(name)


@Registry.register("models", "qwen3")
class Qwen3Model:
    pass


@Registry.register("tuners", "QLoRA")
class QLoRATuner:
    pass
