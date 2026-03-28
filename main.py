import torch
import pandas as pd

from src.data_processing.metadata import create_metadata_csv
from src.data_processing.transforms import Transform
from src.data_processing.stats import compute_and_save_stats

from src.modeling.dataset import Dataset, get_dataloaders
from src.modeling.model import get_renaissance_model
from src.modeling.trainer import Trainer


# PARAMS
DATA_DIRS = {
        "accidental": {"reddit": "data/raw/reddit_data"},
        "paintings": {
            "wikiart": "data/raw/wikiart_data",
            "met": "data/raw/met_data",
            "rijks": "data/raw/rijks_data"
        }
    }

BALANCING_METHOD = "none"
# options: "none", "weighted_loss", "sampler"

EPOCHS = 10


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

    train_loader, val_loader = get_dataloaders(
        df,
        train_transform,
        val_transform,
        use_sampler=use_sampler
    )

        
    print("\n Getting model")
    model = get_renaissance_model(num_classes=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if BALANCING_METHOD == "weighted_loss":
        class_counts = df["label"].value_counts()

        counts = torch.tensor([
            class_counts.get("normal", 1),
            class_counts.get("accidental", 1),
            class_counts.get("paintings", 1)
        ], dtype=torch.float)

        weights = 1.0 / counts
        weights = weights / weights.sum()
        weights = weights.to(device)

        print("\n Using Weighted Loss:", weights)

        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    else:
        print("\n Using standard CrossEntropyLoss")
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    print("\n Training model")
    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch+1}/{EPOCHS}")

        loss = trainer.train_epoch()
        acc = trainer.validate()

        print(f"Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

    print("\n Training complete!")


if __name__ == "__main__":
    main()