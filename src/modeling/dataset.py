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
        return None

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

        if self.label_map:
            label = self.label_map[row["label"]]
        else:
            label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def build_sampler(train_df):

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

    if task == "binary":
        if binary_setup is None:
            raise ValueError("binary_setup must be provided for binary task")

        pos = binary_setup["positive"]
        neg = binary_setup["negative"]

        df = df[df["label"].isin(pos + neg)].copy()

        df["label"] = df["label"].apply(lambda x: 1 if x in pos else 0)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )


    label_map = None if task == "binary" else get_label_map(task)


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
    print("DUPA_5")

    if balancing == "sampler":
        sampler = build_sampler(train_df)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )
        print("DUPA_54")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        print("DUPA_52")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    print("DUPA_5")


    return train_loader, val_loader, train_df, val_df