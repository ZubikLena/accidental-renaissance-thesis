import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm

from src.config import load_config
from src.data_pipeline import build_dataloaders, save_split_records
from src.model import build_model


def create_criterion(config: Dict[str, Any], class_weights: Optional[torch.Tensor], device: torch.device) -> nn.Module:
    loss_config = config.get("training", {}).get("loss", {})
    name = loss_config.get("name", "cross_entropy").lower()
    label_smoothing = float(loss_config.get("label_smoothing", 0.0))

    if name == "cross_entropy":
        weight = class_weights.to(device) if class_weights is not None else None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    raise ValueError(f"Unsupported loss function: {name}")


def create_optimizer(config: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    optimizer_config = config.get("training", {}).get("optimizer", {})
    name = optimizer_config.get("name", "adam").lower()
    lr = float(optimizer_config.get("lr", 1e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    parameters = [p for p in model.parameters() if p.requires_grad]

    if not parameters:
        raise RuntimeError("No trainable parameters were found in the model.")

    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")


def accuracy_from_outputs(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = outputs.argmax(dim=1)
    return float((predictions == targets).float().sum().item() / targets.size(0))


def f1_from_outputs(outputs: torch.Tensor, targets: torch.Tensor, average: str = "macro") -> float:
    predictions = outputs.argmax(dim=1)
    return float(
        f1_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average=average,
            zero_division=0,
        )
    )


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    average: str = "macro",
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    steps = 0

    with tqdm(loader, desc="Train", unit="batch", leave=False) as progress:
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())
            steps += 1
            progress.set_postfix(loss=loss.item())

    if steps == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return {
        "loss": running_loss / steps,
        "accuracy": float((preds == targets).float().sum().item() / len(targets)),
        "f1": f1_score(targets.numpy(), preds.numpy(), average=average, zero_division=0),
    }


def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    average: str = "macro",
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    steps = 0

    with torch.no_grad():
        with tqdm(loader, desc="Valid", unit="batch", leave=False) as progress:
            for images, targets in progress:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())
                steps += 1
                progress.set_postfix(loss=loss.item())

    if steps == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return {
        "loss": running_loss / steps,
        "accuracy": float((preds == targets).float().sum().item() / len(targets)),
        "f1": f1_score(targets.numpy(), preds.numpy(), average=average, zero_division=0),
    }


def save_checkpoint(model: nn.Module, output_dir: str, epoch: int, config: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"best_model_epoch_{epoch}.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, model_path)


def run_experiment(config_path: str) -> None:
    config = load_config(config_path)
    training_config = config.get("training", {})
    device_name = training_config.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    data_loaders, class_names, class_weights, split_records = build_dataloaders(config)
    num_classes = len(class_names)
    model = build_model(config, num_classes).to(device)
    optimizer = create_optimizer(config, model)
    criterion = create_criterion(config, class_weights, device)

    print("Data loaders and model built successfully.")

    num_epochs = int(training_config.get("num_epochs", 10))
    output_dir = training_config.get("output_dir", "outputs/models")
    best_metric: Optional[float] = None
    best_model_path: Optional[str] = None

    early_config = training_config.get("early_stopping", {})
    early_enabled = bool(early_config.get("enabled", False))
    early_monitor = early_config.get("monitor", "val_loss")
    early_mode = early_config.get("mode", "min")
    early_patience = int(early_config.get("patience", 3))
    early_min_delta = float(early_config.get("min_delta", 0.0))
    early_wait = 0

    metrics_average = training_config.get("metrics", {}).get("f1_average", "macro")

    print(f"Starting training on device: {device}")
    print(f"Model: {config.get('model', {}).get('name', 'resnet50')} | Classes: {num_classes}")
    print(f"Class mapping: {json.dumps(class_names, indent=2)}")

    split_save = training_config.get("save_split_records", True)
    split_dir = training_config.get("split_output_dir", os.path.join(output_dir, "splits"))
    if split_save:
        save_split_records(split_records["test"], Path(split_dir) / "rq1_test_records.csv")
        print(f"Saved test split records to: {Path(split_dir) / 'rq1_test_records.csv'}")

    print("Starting training loop...")

    def is_improved(current: float, best: Optional[float], mode: str, min_delta: float) -> bool:
        if best is None:
            return True
        if mode == "min":
            return current < best - min_delta
        return current > best + min_delta

    for epoch in range(1, num_epochs + 1):
        print(f"Starting epoch {epoch}/{num_epochs}")
        start_time = time.time()
        train_metrics = train_epoch(model, data_loaders["train"], optimizer, criterion, device, average=metrics_average)
        val_metrics = evaluate_epoch(model, data_loaders["val"], criterion, device, average=metrics_average) if data_loaders.get("val") else {}
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{num_epochs} - {elapsed:.1f}s")
        print(
            f"  train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.4f} f1: {train_metrics['f1']:.4f}"
        )
        if val_metrics:
            print(
                f"  val   loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} f1: {val_metrics['f1']:.4f}"
            )

        if val_metrics:
            monitor_value = val_metrics.get(early_monitor, val_metrics.get("loss", 0.0))
            if is_improved(monitor_value, best_metric, early_mode, early_min_delta):
                best_metric = monitor_value
                save_checkpoint(model, output_dir, epoch, config)
                best_model_path = os.path.join(output_dir, f"best_model_epoch_{epoch}.pt")
                early_wait = 0
            elif early_enabled:
                early_wait += 1
                if early_wait >= early_patience:
                    print(
                        f"Early stopping triggered after {epoch} epochs. Best {early_monitor}: {best_metric:.4f}."
                    )
                    break

    if data_loaders.get("test"):
        test_metrics = evaluate_epoch(model, data_loaders["test"], criterion, device, average=metrics_average)
        print(
            f"Test loss: {test_metrics['loss']:.4f} acc: {test_metrics['accuracy']:.4f} f1: {test_metrics['f1']:.4f}"
        )

    if best_model_path:
        print(f"Saved best checkpoint to: {best_model_path}")
    else:
        print("Training completed without validation or no improved checkpoint.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classification model on raw image folders using configurable data and balancing.")
    parser.add_argument("config", nargs="?", default="configs/exp.yaml", help="Path to experiment configuration YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
