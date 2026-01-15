from abc import ABC, abstractmethod


class BaseTuner(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def apply(self, model):
        """injection LoRA/QLoRA or Freeze config"""
        pass

    @abstractmethod
    def wrap_for_training(self, model):
        """
        something for RL training... idk:))
        """
        pass
