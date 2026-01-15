import yaml

def load_llm_config(llm):
    with open("configs/llm_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f)
    return llm_config[llm]
    