import sys
from src.train import run_experiment


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp.yaml"
    run_experiment(config_path)


if __name__ == "__main__":
    main()