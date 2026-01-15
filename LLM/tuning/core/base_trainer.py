from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, model, args, train_dataset, eval_dataset=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def compute_loss(self):
        """ (maybe)
        - SFT: CrossEntropy Loss
        - DPO: Log-sigmoid Loss của Chosen vs Rejected
        - GRPO: Advantage-based Loss
        """
        pass