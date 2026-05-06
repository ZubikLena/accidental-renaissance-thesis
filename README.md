## Datasets Used

This project combines multiple sources to study Renaissance-style visual patterns:

### 1. The Metropolitan Museum of Art (MET)
- Source: https://metmuseum.github.io/
- Description: Open-access dataset of artworks
- Used for: Renaissance paintings (filtered by date, medium, and department)
- Size: 586 images

---

### 2. WikiArt
- Source: https://www.wikiart.org/
- Description: Online collection of artworks by style and movement
- Used for: Curated Renaissance paintings (Early, High, Northern, Mannerism)
- Size: 6192 images

---

### 3. Rijksmuseum
- Source: https://data.rijksmuseum.nl/
- Description: Cultural heritage collection with API access
- Used for: European paintings from the Renaissance period
- Size: 87 images

---

### 4. Reddit — r/AccidentalRenaissance
- Source: https://www.reddit.com/r/AccidentalRenaissance/
- Description: Modern photographs resembling Renaissance compositions
- Used for: "Accidental Renaissance" dataset (modern domain)
- Size: 1661 images

---

## ⚙️ Reproducing the Dataset

All datasets can be reconstructed using the provided data collection pipeline. (src/data_collection)

## 🧠 Training Pipeline

This repository now includes a configurable classification pipeline for fine-tuning image models on the combined dataset.

- `resnet50` and `vision transformer` (`vit_b_16`) are supported.
- The pipeline reads raw images from `data/raw` and maps dataset folders to classes using `configs/exp.yaml`.
- Augmentation and class balancing are configurable:
  - `none`, `standard`, `auto_augment`
  - balancing strategies: `none`, `downsample`, `oversample`, `class_weight`
- Only the model head is fine-tuned while the backbone is frozen by default.

### Run training

```bash
python main.py configs/exp.yaml
```

The configuration file is `configs/exp.yaml` and includes dataset mapping, augmentation, balancing, optimizer, and loss settings.
