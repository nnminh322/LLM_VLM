class Registry:
    _REGISTRIES = {"models": {}, "datasets": {}, "tuners": {}, "tasks": {}}

    @classmethod
    def register(cls, catergory: str, name: str):
        def decorator(wrapper_class):
            if catergory not in cls._REGISTRIES:
                raise ValueError(f"wrapper_class: {wrapper_class} not in categories!")
            cls._REGISTRIES[catergory][name] = wrapper_class
            return wrapper_class

        return decorator

    @classmethod
    def get(cls, category: str, name: str):
        return cls._REGISTRIES[category].get(name)


# @Registry.register("models", "qwen3")
# class Qwen3Model:
#     pass


# @Registry.register("tuners", "QLoRA")
# class QLoRATuner:
#     pass
