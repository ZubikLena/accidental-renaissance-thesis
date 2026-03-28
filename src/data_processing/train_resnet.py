import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from dataset import PhotoDataset
from transforms import get_transforms

CSV_PATH = "data/processed/dataset.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def main():

    print("Device:", DEVICE)

    # -------- DATA --------
    train_dataset = PhotoDataset(CSV_PATH, "train", get_transforms())
    val_dataset = PhotoDataset(CSV_PATH, "val", get_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Train batches:", len(train_loader))

    # -------- MODEL --------
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(DEVICE)

    # -------- TRAINING --------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0

    # -------- LOOP --------
    for epoch in range(3):

        model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")

        # -------- SAVE BEST MODEL --------
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "src/models/resnet_best.pth")
            print("Saved BEST model")

    # -------- SAVE FINAL MODEL --------
    torch.save(model.state_dict(), "src/models/resnet_last.pth")
    print("💾 Saved FINAL model")


if __name__ == "__main__":
    main()