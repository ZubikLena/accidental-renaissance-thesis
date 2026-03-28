import pandas as pd
import random
from pathlib import Path

RAW = Path("data/raw")
OUTPUT = Path("data/processed/dataset.csv")

SPLIT = (0.7, 0.15, 0.15)

DATA_SOURCES = {
    "renaissance": ["met_data", "rijks_data", "wikiart_data"],
    "accidental": ["reddit_data"],
    "normal": ["coco_data"],
}


def get_files(folders):
    files = []
    for folder in folders:
        files += list((RAW / folder).glob("*.jpg"))
    return files


def split_files(files):
    random.shuffle(files)
    n = len(files)

    train = files[:int(n * SPLIT[0])]
    val = files[int(n * SPLIT[0]):int(n * (SPLIT[0] + SPLIT[1]))]
    test = files[int(n * (SPLIT[0] + SPLIT[1])):]

    return {
        "train": train,
        "val": val,
        "test": test
    }


def create_records(files_dict, label):
    records = []

    for split, files in files_dict.items():
        for f in files:
            records.append((str(f), label, split))

    return records


def build_dataset():

    all_records = []

    for label, folders in DATA_SOURCES.items():

        print(f"Processing: {label}")

        files = get_files(folders)
        split_dict = split_files(files)

        records = create_records(split_dict, label)
        all_records.extend(records)

        print(f"  Total: {len(files)}")

    df = pd.DataFrame(all_records, columns=["path", "label", "split"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("\nDataset saved to:", OUTPUT)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nSplit distribution:")
    print(df["split"].value_counts(normalize=True))


if __name__ == "__main__":
    build_dataset()