from collections import Counter
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, WeightedRandomSampler
from torchvision import transforms

DEFAULT_SOURCE_LABEL_MAP = {
    "coco_data": "ordinary",
    "met_data": "paintings",
    "reddit_data": "accidental",
    "rijks_data": "paintings",
    "wikiart_data": "paintings",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class ImagePathDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], label_to_index: Dict[str, int], transform: Optional[transforms.Compose] = None):
        self.records = records
        self.label_to_index = label_to_index
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = Path(record["path"])
        label = self.label_to_index[record["label"]]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, label


def collect_image_records(raw_root: str, source_label_map: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    raw_root_path = Path(raw_root)
    if not raw_root_path.exists():
        raise FileNotFoundError(f"Raw data root does not exist: {raw_root}")

    label_map = source_label_map or DEFAULT_SOURCE_LABEL_MAP
    records: List[Dict[str, str]] = []

    for source_dir in sorted(raw_root_path.iterdir()):
        if not source_dir.is_dir():
            continue

        source_name = source_dir.name
        label = label_map.get(source_name)
        if label is None:
            continue

        for image_path in sorted(source_dir.rglob("*")):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append({"path": str(image_path), "label": label, "source": source_name})

    if not records:
        raise RuntimeError(f"No image records found under {raw_root}")

    return records


def filter_records_by_labels(records: List[Dict[str, str]], selected_labels: List[str]) -> List[Dict[str, str]]:
    selected = set(selected_labels)
    if not selected:
        return records
    filtered = [record for record in records if record["label"] in selected]
    if not filtered:
        raise ValueError(f"No records found for selected labels: {selected_labels}")
    return filtered


def save_split_records(records: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(records)


def split_records(records: List[Dict[str, str]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("Invalid split ratios.")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test split ratios must sum to 1.0.")

    labels = [record["label"] for record in records]
    if test_ratio > 0:
        train_val_records, rq1_test_records = train_test_split(
            records,
            test_size=test_ratio,
            stratify=labels,
            random_state=seed,
        )
    else:
        train_val_records, rq1_test_records = records, []

    if val_ratio > 0:
        relative_val = val_ratio / (train_ratio + val_ratio)
        train_records, val_records = train_test_split(
            train_val_records,
            test_size=relative_val,
            stratify=[record["label"] for record in train_val_records],
            random_state=seed,
        )
    else:
        train_records, val_records = train_val_records, []

    return train_records, val_records, rq1_test_records


def build_label_index(records: Iterable[Dict[str, str]]) -> Dict[str, int]:
    labels = sorted({record["label"] for record in records})
    return {label: index for index, label in enumerate(labels)}


def create_transforms(config: Dict[str, Any], is_train: bool = True) -> transforms.Compose:
    augmentation_config = config.get("augmentation", {})
    strategy = augmentation_config.get("strategy", "none").lower()
    image_size = config.get("image_size", 224)
    resize_size = augmentation_config.get("resize", image_size)

    common: List[Any] = [transforms.Resize(resize_size)]

    if is_train and strategy != "none":
        crop_size = augmentation_config.get("crop", image_size)
        horizontal_flip = augmentation_config.get("horizontal_flip", True)
        rotation = augmentation_config.get("rotation", 0)
        color_jitter = augmentation_config.get("color_jitter", {})
        auto_augment = augmentation_config.get("auto_augment", False)

        common = [transforms.RandomResizedCrop(crop_size)]
        if horizontal_flip:
            common.append(transforms.RandomHorizontalFlip())
        if rotation:
            common.append(transforms.RandomRotation(rotation))
        if color_jitter:
            brightness = color_jitter.get("brightness", 0.0)
            contrast = color_jitter.get("contrast", 0.0)
            saturation = color_jitter.get("saturation", 0.0)
            hue = color_jitter.get("hue", 0.0)
            common.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
        if auto_augment:
            common.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    else:
        common.append(transforms.CenterCrop(image_size))

    common.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])
    return transforms.Compose(common)


def downsample_records(records: List[Dict[str, str]], seed: int) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        grouped.setdefault(record["label"], []).append(record)

    min_count = min(len(items) for items in grouped.values())
    balanced: List[Dict[str, str]] = []
    rng = torch.Generator()
    rng.manual_seed(seed)

    for label, items in grouped.items():
        if len(items) <= min_count:
            balanced.extend(items)
        else:
            indices = torch.randperm(len(items), generator=rng)[:min_count].tolist()
            balanced.extend([items[i] for i in indices])

    return balanced


def build_sampler(records: List[Dict[str, str]], label_to_index: Dict[str, int], strategy: str) -> Optional[Sampler]:
    strategy = strategy.lower()
    if strategy != "oversample":
        return None

    label_counts = Counter(record["label"] for record in records)
    if not label_counts:
        return None

    class_counts = {label_to_index[label]: count for label, count in label_counts.items()}
    max_class_count = max(class_counts.values())
    sample_weights = [1.0 / class_counts[label_to_index[record["label"]]] for record in records]
    num_samples = max_class_count * len(class_counts)
    return WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)


def compute_class_weights(records: List[Dict[str, str]], label_to_index: Dict[str, int]) -> torch.Tensor:
    counts = Counter(record["label"] for record in records)
    num_classes = len(label_to_index)
    total = sum(counts.values())
    weights = [0.0] * num_classes

    for label, index in label_to_index.items():
        count = counts[label]
        if count > 0:
            weights[index] = float(total) / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(config: Dict[str, Any]) -> Tuple[Dict[str, DataLoader], Dict[str, int], Optional[torch.Tensor]]:
    raw_root = config.get("raw_root") or config.get("dataset", {}).get("raw_root") or "data/raw"
    source_label_map = config.get("source_label_map") or config.get("dataset", {}).get("source_label_map")
    split_config = config.get("split", {})
    train_ratio = float(split_config.get("train", 0.8))
    val_ratio = float(split_config.get("val", 0.1))
    test_ratio = float(split_config.get("test", 0.1))
    seed = int(config.get("seed", 42))
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    num_workers = int(config.get("training", {}).get("num_workers", 4))
    balancing = config.get("balancing", {}).get("strategy", "none").lower()

    records = collect_image_records(raw_root, source_label_map)
    selected_labels = config.get("selected_labels", [])
    if selected_labels:
        records = filter_records_by_labels(records, selected_labels)

    train_records, val_records, rq1_test_records = split_records(records, train_ratio, val_ratio, test_ratio, seed)

    if balancing == "downsample":
        train_records = downsample_records(train_records, seed)

    label_to_index = build_label_index(records)
    class_weights = None
    if balancing == "class_weight":
        class_weights = compute_class_weights(train_records, label_to_index)

    train_transform = create_transforms(config.get("augmentation", {}), is_train=True)
    eval_transform = create_transforms(config.get("augmentation", {}), is_train=False)

    train_dataset = ImagePathDataset(train_records, label_to_index, transform=train_transform)
    val_dataset = ImagePathDataset(val_records, label_to_index, transform=eval_transform) if val_records else None
    test_dataset = ImagePathDataset(rq1_test_records, label_to_index, transform=eval_transform) if rq1_test_records else None

    train_sampler = build_sampler(train_records, label_to_index, balancing)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        if test_dataset is not None
        else None
    )

    class_names = {index: label for label, index in label_to_index.items()}
    return (
        {"train": train_loader, "val": val_loader, "test": test_loader},
        class_names,
        class_weights,
        {"train": train_records, "val": val_records, "test": rq1_test_records},
    )
