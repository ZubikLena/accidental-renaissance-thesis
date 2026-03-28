import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from dataset_paintings import PaintingDataset
from transforms import get_transforms

CSV_PATH = "data/processed/dataset.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    # -------- DATA --------
    train_dataset = PaintingDataset(CSV_PATH, "train", get_transforms())
    val_dataset = PaintingDataset(CSV_PATH, "val", get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # -------- MODEL --------
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(DEVICE)

    # -------- TRAINING --------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0

    for epoch in range(2):

        model.train()

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "models/resnet_painting_pretrained.pth")
            print("✅ Saved painting-pretrained model")


if __name__ == "__main__":
    main()