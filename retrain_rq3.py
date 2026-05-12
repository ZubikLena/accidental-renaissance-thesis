import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.data_pipeline import (
    ImagePathDataset,
    build_label_index,
    save_split_records,
)
from src.model import build_model
from src.train import train_epoch, evaluate_epoch


# ==========================================
# CONFIG

CHECKPOINT_PATH = "outputs/models/vit_oversample.pt"
RQ1_TEST_PATH = "outputs/splits/rq1_test_records.csv"
METADATA_PATH = "data/metadata.csv"

MODEL_TYPE = "vit_b_16"   # resnet50 / vit_b_16
TRANSFER_MODE = "reset_head"  # reset_head / keep_head

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42
IMAGE_SIZE = 224

OUTPUT_DIR = "outputs/models"
SPLIT_DIR = "outputs/splits"
ORDINARY_LABELS = ["normal"]
PAINTING_LABELS = ["paintings"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent


def normalize_path(path: str) -> str:
    path = str(path).replace("\\", "/")

    if "data/" in path:
        return path[path.index("data/"):]

    return path


# ==========================================

def load_rq1_test_paths():
    df = pd.read_csv(RQ1_TEST_PATH)

    ordinary_df = df[df["label"].isin(["ordinary", "normal"])]

    paths = {
        normalize_path(p)
        for p in ordinary_df["path"].tolist()
    }

    print(f"Loaded {len(paths)} ordinary test images from RQ1")

    return paths


def load_metadata():
    df = pd.read_csv(METADATA_PATH)

    records = df.to_dict(orient="records")

    return records


def create_rq3_splits(all_records):
    rq1_test_paths = load_rq1_test_paths()

    ordinary_records = [
        r for r in all_records
        if r["label"] in ORDINARY_LABELS
    ]

    painting_records = [
        r for r in all_records
        if r["label"] in PAINTING_LABELS
    ]

    print("\nSample RQ1 paths:")
    print(list(rq1_test_paths)[:5])

    print("\nSample metadata paths:")
    print([
        normalize_path(r["path"])
        for r in ordinary_records[:5]
    ])

    rq3_test_ordinary = [
        r for r in ordinary_records
        if normalize_path(r["path"]) in rq1_test_paths
    ]

    train_val_ordinary = [
        r for r in ordinary_records
        if normalize_path(r["path"]) not in rq1_test_paths
    ]

    print(f"Train/Val ordinary: {len(train_val_ordinary)}")
    print(f"RQ3 test ordinary: {len(rq3_test_ordinary)}")

    test_paintings = painting_records[:len(rq3_test_ordinary)]

    used_test_paths = {
        r["path"] for r in test_paintings
    }

    remaining_paintings = [
        r for r in painting_records
        if r["path"] not in used_test_paths
    ]

    rq3_test = rq3_test_ordinary + test_paintings

    train_val_pool = (
        train_val_ordinary
        + remaining_paintings
    )

    labels = [
        r["label"]
        for r in train_val_pool
    ]

    train_records, val_records = train_test_split(
        train_val_pool,
        test_size=0.1,
        stratify=labels,
        random_state=SEED
    )

    print("\nFinal split sizes:")
    print(f"Train: {len(train_records)}")
    print(f"Val: {len(val_records)}")
    print(f"Test: {len(rq3_test)}")

    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
        ("test", rq3_test),
    ]:
        counts = Counter(
            r["label"]
            for r in split_records
        )
        print(f"{split_name}: {dict(counts)}")

    return train_records, val_records, rq3_test


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # use original training config
    original_config = checkpoint.get("config")

    if original_config is None:
        raise ValueError(
            "Checkpoint does not contain config. "
            "Cannot safely rebuild architecture."
        )

    print("\nLoaded original config from checkpoint:")
    print(original_config["model"])

    model = build_model(
        original_config,
        num_classes=2
    ).to(DEVICE)

    # load exact trained weights
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )

    print(f"\nLoaded checkpoint: {CHECKPOINT_PATH}")

    if missing:
        print(f"Missing keys: {missing}")

    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    # -----------------------------------
    # B1 → RESET HEAD
    # -----------------------------------
    if TRANSFER_MODE == "reset_head":
        print("\nTransfer mode: RESET HEAD")

        if MODEL_TYPE == "resnet50":
            old_head = model.fc

            if isinstance(old_head, nn.Sequential):
                first_linear = None

                for layer in old_head:
                    if isinstance(layer, nn.Linear):
                        first_linear = layer
                        break

                if first_linear is None:
                    raise ValueError("No Linear layer found in ResNet head")

                in_features = first_linear.in_features

            else:
                in_features = model.fc.in_features

            print(f"Resetting ResNet head: {in_features} -> 2")

            model.fc = nn.Linear(in_features, 2).to(DEVICE)

        elif MODEL_TYPE == "vit_b_16":
            old_head = model.heads.head

            if isinstance(old_head, nn.Sequential):
                first_linear = None

                for layer in old_head:
                    if isinstance(layer, nn.Linear):
                        first_linear = layer
                        break

                if first_linear is None:
                    raise ValueError(
                        "No Linear layer found in ViT head"
                    )

                in_features = first_linear.in_features

            else:
                in_features = model.heads.head.in_features

            print(f"Resetting ViT head: {in_features} -> 2")

            model.heads.head = nn.Linear(in_features, 2).to(DEVICE)

    # -----------------------------------
    # B2 → KEEP HEAD
    # -----------------------------------
    elif TRANSFER_MODE == "keep_head":
        print("\nTransfer mode: KEEP HEAD")

    else:
        raise ValueError(
            "TRANSFER_MODE must be reset_head or keep_head"
        )

    return model


# ==========================================
# DATALOADERS
# ==========================================

def build_dataloaders(
    train_records,
    val_records,
    test_records
):
    all_records = (
        train_records
        + val_records
        + test_records
    )

    label_to_index = build_label_index(
        all_records
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_ds = ImagePathDataset(
        train_records,
        label_to_index,
        train_transform
    )

    val_ds = ImagePathDataset(
        val_records,
        label_to_index,
        eval_transform
    )

    test_ds = ImagePathDataset(
        test_records,
        label_to_index,
        eval_transform
    )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True
        ),
        "val": torch.utils.data.DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False
        ),
        "test": torch.utils.data.DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    }

    return loaders


# ==========================================
# TRAINING
# ==========================================

def train():
    all_records = load_metadata()

    train_records, val_records, test_records = create_rq3_splits(
        all_records
    )

    split_output_dir = ROOT_DIR / SPLIT_DIR
    os.makedirs(
        split_output_dir,
        exist_ok=True
    )

    clean_test_records = [
        {
            "path": r["path"],
            "label": r["label"],
            "source": r.get("source", "")
        }
        for r in test_records
    ]

    save_split_records(
        clean_test_records,
        split_output_dir / "rq3_test_records.csv"
    )
    print(f"Saved RQ3 test split to: {split_output_dir / 'rq3_test_records.csv'}")

    loaders = build_dataloaders(
        train_records,
        val_records,
        test_records
    )

    model = load_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_path = None

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        start = time.time()

        train_metrics = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            DEVICE,
            average="macro"
        )

        val_metrics = evaluate_epoch(
            model,
            loaders["val"],
            criterion,
            DEVICE,
            average="macro"
        )

        print(
            f"Train F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            model_output_dir = ROOT_DIR / OUTPUT_DIR
            os.makedirs(
                model_output_dir,
                exist_ok=True
            )

            best_model_path = os.path.join(
                model_output_dir,
                f"rq3_{TRANSFER_MODE}_best.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict()
                },
                best_model_path
            )

            print("✓ Saved new best model")

        print(
            f"Epoch time: {time.time()-start:.1f}s"
        )

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nFinal test evaluation:")
    test_metrics = evaluate_epoch(
        model,
        loaders["test"],
        criterion,
        DEVICE,
        average="macro"
    )

    print(test_metrics)


if __name__ == "__main__":
    train()