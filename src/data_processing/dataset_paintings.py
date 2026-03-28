import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PaintingDataset(Dataset):

    def __init__(self, csv_file, split, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.sample(3000)

        self.df = self.df[self.df["split"] == split]

        self.transform = transform

        # 2 classes:
        # 0 = not painting
        # 1 = painting
        self.label_map = {
            "renaissance": 1,
            "accidental": 0,
            "normal": 0
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