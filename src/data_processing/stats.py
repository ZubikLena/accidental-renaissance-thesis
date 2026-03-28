import torch
import json
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_and_save_stats(dataset, cache_path="outputs/dataset_stats.json"):

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    mean = 0.0
    std = 0.0
    total = 0

    for images, _ in tqdm(loader):
        batch = images.size(0)
        images = images.view(batch, images.size(1), -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += batch

    stats = {
        "mean": (mean / total).tolist(),
        "std": (std / total).tolist()
    }

    os.makedirs("outputs", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(stats, f)

    return stats