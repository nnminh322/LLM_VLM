import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_llm_config(llm_name):
    with open("configs/model_config/llm_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f)
    return llm_config[llm_name]


def load_sft_config(sft_config_name):
    with open("configs/task_config/sft_config", "r") as f:
        sft_config = yaml.safe_load(f)
    return sft_config[sft_config_name]
