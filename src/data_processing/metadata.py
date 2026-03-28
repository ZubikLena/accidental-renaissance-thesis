import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def create_metadata_csv(data_dirs, output_path="data/metadata.csv"):
    records = []
    valid_ext = {".jpg", ".jpeg", ".png"}

    print("Scanning images...")

    for label, sources in data_dirs.items():
        for source_name, folder in sources.items():

            for path in Path(folder).rglob("*"):
                if path.suffix.lower() in valid_ext:

                    records.append({
                        "path": str(path.absolute()),
                        "label": label,
                        "source": source_name
                    })

    if not records:
        raise ValueError("No images found!")

    df = pd.DataFrame(records)

    print("Extracting image sizes...")

    widths, heights = [], []
    for path in tqdm(df["path"]):
        try:
            with Image.open(path) as img:
                w, h = img.size
            widths.append(w)
            heights.append(h)
        except:
            widths.append(None)
            heights.append(None)

    df["width"] = widths
    df["height"] = heights
    df = df.dropna()

    Path("data").mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved metadata → {output_path}")
    return df