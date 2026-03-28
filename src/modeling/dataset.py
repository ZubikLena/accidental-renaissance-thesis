import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split


class Dataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_map = {"normal": 0, "accidental": 1, "paintings": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            image = Image.open(row["path"]).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))

        label = self.label_map[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(df, train_transform, val_transform, batch_size=32, use_sampler=False):

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"]
    )

    train_dataset = Dataset(train_df, transform=train_transform)
    val_dataset = Dataset(val_df, transform=val_transform)

    if use_sampler:
        class_counts = train_df["label"].value_counts()

        weights = train_df["label"].map(
            lambda x: 1.0 / class_counts[x]
        ).values

        sampler = WeightedRandomSampler(weights, num_samples=len(weights))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader