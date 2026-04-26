import torch
import os
import json
from datetime import datetime

from src.data_processing.metadata import create_metadata_csv
from src.data_processing.transforms import Transform
from src.data_processing.stats import compute_and_save_stats

from src.modeling.dataset import Dataset, get_dataloaders
from src.modeling.model import get_model
from src.modeling.trainer import Trainer
from src.modeling.utils.config_loader import load_config
from src.modeling.utils.criterion import get_criterion

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
BINARY_SETUP = CONFIG.get("binary_setup", None)


def main():

    df = create_metadata_csv(DATA_DIRS)

    print("\n Class distribution:")
    print(df["label"].value_counts())

    basic_transform = Transform.get_basic_transform()

    dataset = Dataset(df, transform=basic_transform)
    stats = compute_and_save_stats(dataset)

    train_transform = Transform.get_calibrated_transform(
        stats["mean"], stats["std"], train=True
    )

    val_transform = Transform.get_calibrated_transform(
        stats["mean"], stats["std"], train=False
    )

    use_sampler = (BALANCING_METHOD == "sampler")
    print("\nDF SIZE BEFORE DATALOADER:", len(df))

    print("DEBUG: About to call get_dataloaders")
    train_loader, val_loader, train_df, val_df = get_dataloaders(
        df,
        train_transform,
        val_transform,
        BATCH_SIZE,
        task=TASK,
        balancing=use_sampler,
        binary_setup=BINARY_SETUP
    )
    print("DEBUG: get_dataloaders returned successfully")

    print("DEBUG: About to print label counts")
    print(train_df["label"].value_counts())
    print(val_df["label"].value_counts())
    print("DEBUG: Finished printing label counts")

    print("\n Getting model")
    model = get_model(model_name=MODEL_NAME, num_classes=2 if TASK == "binary" else 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = get_criterion(df, TASK, BALANCING_METHOD, device, binary_setup=BINARY_SETUP)

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

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    
    print("\n Training complete!")


if __name__ == "__main__":
    main()