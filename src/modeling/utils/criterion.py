import torch

def get_criterion(df, task, balancing, device, binary_setup=None):

    if balancing == "weighted_loss":
        class_counts = df["label"].value_counts()

        if task == "binary":
            if binary_setup is None:
                raise ValueError("binary_setup must be provided for binary weighted loss")

            # If labels are already 0/1, use them directly.
            if set(class_counts.index).issubset({0, 1}):
                counts = torch.tensor([
                    class_counts.get(0, 1),
                    class_counts.get(1, 1)
                ], dtype=torch.float)
            else:
                labels = df["label"].apply(
                    lambda x: 1 if x in binary_setup["positive"] else 0
                )
                class_counts = labels.value_counts()
                counts = torch.tensor([
                    class_counts.get(0, 1),
                    class_counts.get(1, 1)
                ], dtype=torch.float)

        else:
            counts = torch.tensor([
                class_counts.get(0, 1),
                class_counts.get(1, 1),
                class_counts.get(2, 1)
            ], dtype=torch.float)

        weights = 1.0 / counts
        weights = weights / weights.sum()
        weights = weights.to(device)

        print("\n Using Weighted Loss:", weights)

        return torch.nn.CrossEntropyLoss(weight=weights)

    else:
        print("\n Using standard CrossEntropyLoss")
        return torch.nn.CrossEntropyLoss()