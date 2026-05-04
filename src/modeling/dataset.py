import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split

class Dataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            image = Image.open(row["path"]).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def build_sampler(train_df):
    labels = train_df["label"]
    class_counts = labels.value_counts()
    weights = 1.0 / labels.map(class_counts)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def get_dataloaders(
    df,
    train_transform,
    val_transform,
    batch_size,
    task,
    balancing,
    binary_setup
):


    if task == "binary":
        pos = binary_setup["positive"]
        neg = binary_setup["negative"]

        df = df[df["label"].isin(pos + neg)].copy()
        df["label"] = df["label"].apply(lambda x: 1 if x in pos else 0)

    elif task == "multiclass":
        label_map = {
            "normal": 0,
            "accidental": 1,
            "paintings": 2
        }

        df["label"] = df["label"].str.strip().str.lower()
        df["label"] = df["label"].map(label_map)

        if df["label"].isnull().any():
            raise ValueError("Some labels could not be mapped to integers")

    else:
        raise ValueError(f"Unknown task: {task}")


    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    )

    train_dataset = Dataset(train_df, transform=train_transform)
    val_dataset = Dataset(val_df, transform=val_transform)
    test_dataset = Dataset(test_df, transform=val_transform)  # Use val_transform for test, no augmentation

    if balancing == "sampler":
        sampler = build_sampler(train_df)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df