import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PhotoDataset(Dataset):

    def __init__(self, csv_file, split, transform=None):
        self.df = pd.read_csv(csv_file)

        self.df = self.df[self.df["split"] == split]

        self.df = self.df[self.df["label"].isin(["accidental", "normal"])]

        self.transform = transform

        self.label_map = {
            "accidental": 0,
            "normal": 1
        }

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")
        label = self.label_map[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label