from abc import ABC, abstractmethod
from typing import List, Any, Dict
from torch.utils.data import Dataset


class BaseData(Dataset, ABC):
    def __init__(self, data_path: str, processor: Any, config: Dict):
        """_summary_

        Args:
            data_path (str)
            processor (Any): Object for process each record. Maybe tokenizer for LLM or CNN_decoder for VLM
            config (Dict)
        """
        self.data_path = data_path
        self.processor = processor
        self.config = config
        self.data = self._load_raw_data()

    @abstractmethod
    def _load_raw_data(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        SFT: Trả về {'input_ids', 'labels'}
        DPO: Trả về {'prompt', 'chosen', 'rejected'}
        GRPO: Trả về {'prompt', 'reward_func_metadata'}
        """
        pass

    @abstractmethod
    def get_collator(self):
        pass

    def __len__(self):
        return len(self.data)  # type: ignore
