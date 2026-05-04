import torch
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.data_processing.metadata import create_metadata_csv
from src.data_processing.transforms import Transform

from src.modeling.dataset import Dataset, get_dataloaders
from src.modeling.model import get_model
from src.modeling.trainer import Trainer
from src.modeling.utils.config_loader import load_config
from src.modeling.utils.criterion import get_criterion

# ImageNet normalization stats for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# PARAMS
DATA_DIRS = {
        "accidental": {"reddit": "data/raw/reddit_data"},
        "paintings": {
            "wikiart": "data/raw/wikiart_data",
            "met": "data/raw/met_data",
            "rijks": "data/raw/rijks_data"
        },
        "normal": {"coco": "data/raw/coco_data"}
        
    }


CONFIG = load_config("configs/exp.yaml")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

exp_name = f"{CONFIG['model_name']}_{CONFIG['task']}_{CONFIG['balancing']}_{timestamp}"

output_dir = os.path.join("outputs", exp_name)
os.makedirs(output_dir, exist_ok=True)

MODEL_NAME = CONFIG["model_name"]
TASK = CONFIG["task"]
BALANCING_METHOD = CONFIG["balancing"]

training = CONFIG["training"]
EPOCHS = training["epochs"]
BATCH_SIZE = training["batch_size"]
LR = training["lr"]
EARLY_STOPPING_PATIENCE = training.get("early_stopping_patience", None)
EARLY_STOPPING_MIN_DELTA = training.get("early_stopping_min_delta", 0.0)
BINARY_SETUP = CONFIG.get("binary_setup", None)

GLOBAL_SPLIT_PATH = os.path.join("outputs", f"global_splits_{TASK}.json")


def load_or_create_global_split(df, split_path, task, binary_setup, val_ratio=0.15, test_ratio=0.15, random_state=42):
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    if "label" not in df.columns:
        raise ValueError("Dataframe must include a 'label' column for global split generation")

    def apply_task_filter(subset_df):
        if task == "binary":
            pos = binary_setup["positive"]
            neg = binary_setup["negative"]

            subset_df = subset_df[subset_df["label"].isin(pos + neg)].copy()
            subset_df["label"] = subset_df["label"].apply(lambda x: 1 if x in pos else 0)

        elif task == "multiclass":
            label_map = {
                "normal": 0,
                "accidental": 1,
                "paintings": 2
            }

            subset_df["label"] = subset_df["label"].str.strip().str.lower()
            subset_df["label"] = subset_df["label"].map(label_map)

            if subset_df["label"].isnull().any():
                raise ValueError("Some labels could not be mapped to integers")

        else:
            raise ValueError(f"Unknown task: {task}")

        return subset_df

    filtered_df = apply_task_filter(df.copy())

    train_df, temp_df = train_test_split(
        filtered_df,
        test_size=val_ratio + test_ratio,
        stratify=filtered_df["label"],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df["label"],
        random_state=random_state
    )

    split_indices = {
        "train": train_df.index.tolist(),
        "val": val_df.index.tolist(),
        "test": test_df.index.tolist()
    }

    with open(split_path, "w") as f:
        json.dump(split_indices, f)

    return split_indices


def main():

    df = create_metadata_csv(DATA_DIRS)

    print("\n Class distribution:")
    print(df["label"].value_counts())

    train_transform = Transform.get_calibrated_transform(
        IMAGENET_MEAN, IMAGENET_STD, train=True
    )

    val_transform = Transform.get_calibrated_transform(
        IMAGENET_MEAN, IMAGENET_STD, train=False
    )

    print(f"\nDF SIZE BEFORE DATALOADER: {len(df)}")
    print(f"Using balancing method: {BALANCING_METHOD}")

    global_split_indices = load_or_create_global_split(df, GLOBAL_SPLIT_PATH, TASK, BINARY_SETUP)
    print(f"Using global test split from: {GLOBAL_SPLIT_PATH}")

    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_dataloaders(
        df,
        train_transform,
        val_transform,
        BATCH_SIZE,
        task=TASK,
        balancing=BALANCING_METHOD,
        binary_setup=BINARY_SETUP,
        global_split_indices=global_split_indices
    )

    print("\nData splits:")
    print(f"Train set ({len(train_df)} samples):")
    print(train_df["label"].value_counts())

    print(f"\nVal set ({len(val_df)} samples):")
    print(val_df["label"].value_counts())

    print(f"\nTest set ({len(test_df)} samples):")
    print(test_df["label"].value_counts())

    train_df.to_csv(os.path.join("outputs/train_split.csv"), index=False)
    val_df.to_csv(os.path.join("outputs/val_split.csv"), index=False)
    test_df.to_csv(os.path.join("outputs/test_split.csv"), index=False)
    print(f"\nSaved split details to outputs/")

    print("\n Getting model")
    model = get_model(model_name=MODEL_NAME, num_classes=2 if TASK == "binary" else 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = get_criterion(train_df, TASK, BALANCING_METHOD, device, binary_setup=BINARY_SETUP)

    patience = EARLY_STOPPING_PATIENCE
    min_delta = EARLY_STOPPING_MIN_DELTA
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if patience is not None:
        print(f"Using early stopping with patience={patience}, min_delta={min_delta}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    print("\n Training model")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch+1}/{EPOCHS}")

        train_loss = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pt")
            )
            print("Saved best model")

        if patience is not None:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Early stopping counter: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    
    print("\n Training complete!")


if __name__ == "__main__":
    main()