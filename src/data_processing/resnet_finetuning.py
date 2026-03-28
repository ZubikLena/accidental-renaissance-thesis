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

os.makedirs("models", exist_ok=True)


def main():

    print("Device:", DEVICE)

    # -------- DATA --------
    train_dataset = PhotoDataset(CSV_PATH, "train", get_transforms())
    val_dataset = PhotoDataset(CSV_PATH, "val", get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    # -------- MODEL --------
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 🔥 LOAD PRETRAINED PAINTING WEIGHTS
    model.load_state_dict(torch.load("models/resnet_painting_pretrained.pth"))

    model.to(DEVICE)

    # -------- TRAINING --------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # lower LR for fine-tuning

    best_val_acc = 0

    # -------- LOOP --------
    for epoch in range(3):

        model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

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
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")

        # -------- SAVE BEST --------
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "models/resnet_finetuned.pth")
            print("✅ Saved fine-tuned model")

    print("Done fine-tuning")


if __name__ == "__main__":
    main()