class Registry:
    _REGISTRIES = {"models": {}, "datasets": {}, "tuners": {}, "tasks": {}}

    @classmethod
    def register(cls, category: str, name: str):
        def decorator(wrapper_class):
            if category not in cls._REGISTRIES:
                raise ValueError(f"wrapper_class: {wrapper_class} not in categories!")
            cls._REGISTRIES[category][name] = wrapper_class
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
