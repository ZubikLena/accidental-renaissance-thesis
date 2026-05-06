from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a mapping at the top level.")

    experiment = config.get("experiment")
    if isinstance(experiment, dict):
        config = {**config, **experiment}
        config.pop("experiment", None)

    return config
