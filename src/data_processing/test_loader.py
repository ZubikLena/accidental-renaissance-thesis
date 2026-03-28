from torch.utils.data import DataLoader

from dataset import PhotoDataset
from transforms import get_transforms

CSV_PATH = "data/processed/dataset.csv"


def main():

    dataset = PhotoDataset(
        csv_file=CSV_PATH,
        split="train",
        transform=get_transforms()
    )

    print("Dataset size:", len(dataset))

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for images, labels in loader:
        print("Images shape:", images.shape)
        print("Labels:", labels)
        break


if __name__ == "__main__":
    main()