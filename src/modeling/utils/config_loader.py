import yaml
import os


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    required_keys = ["model_name", "training"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing key '{key}' in config: {path}")

    training_keys = ["epochs", "batch_size", "lr"]

    for key in training_keys:
        if key not in config["training"]:
            raise ValueError(f"Missing training key '{key}' in config: {path}")

    return config