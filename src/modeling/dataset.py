import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split


def get_label_map(task, binary_setup=None):

    if task == "multiclass":
        return {
            "normal": 0,
            "accidental": 1,
            "paintings": 2
        }

    elif task == "binary":

        if binary_setup is None:
            raise ValueError("binary_setup must be provided for binary task")

        label_map = {}

        for cls in binary_setup["positive"]:
            label_map[cls] = 1

        for cls in binary_setup["negative"]:
            label_map[cls] = 0

        return label_map

    else:
        raise ValueError(f"Unknown task: {task}")

class Dataset(Dataset):

    def __init__(self, dataframe, transform=None, label_map=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

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



def build_sampler(train_df, task, binary_setup=None):

    if task == "binary":
        labels = train_df["label"].apply(
            lambda x: 1 if x in binary_setup["positive"] else 0
        )
    else:
        labels = train_df["label"]

    class_counts = labels.value_counts()

    weights = labels.map(lambda x: 1.0 / class_counts[x]).values

    return WeightedRandomSampler(weights, num_samples=len(weights))


def get_dataloaders(
    df,
    train_transform,
    val_transform,
    batch_size=32,
    task="multiclass",
    balancing="none",
    binary_setup=None
):

    # Split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"]
    )

    # Label mapping
    label_map = get_label_map(task, binary_setup=binary_setup)

    # Datasets
    train_dataset = Dataset(
        train_df,
        transform=train_transform,
        label_map=label_map
    )

    val_dataset = Dataset(
        val_df,
        transform=val_transform,
        label_map=label_map
    )

    # Sampler
    if balancing == "sampler":
        sampler = build_sampler(train_df, task)

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