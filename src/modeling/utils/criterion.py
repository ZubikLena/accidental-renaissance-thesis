import torch

def get_criterion(df, task, balancing, device, binary_setup=None):

    if balancing == "weighted_loss":

        if task == "binary":
            labels = df["label"].apply(
                lambda x: 1 if x in binary_setup["positive"] else 0
            )
            class_counts = labels.value_counts()

            counts = torch.tensor([
                class_counts.get(0, 1),
                class_counts.get(1, 1)
            ], dtype=torch.float)

        else:
            class_counts = df["label"].value_counts()
            counts = torch.tensor([
                class_counts.get("normal", 1),
                class_counts.get("accidental", 1),
                class_counts.get("paintings", 1)
            ], dtype=torch.float)

        weights = 1.0 / counts
        weights = weights / weights.sum()
        weights = weights.to(device)

        print("\n Using Weighted Loss:", weights)

        return torch.nn.CrossEntropyLoss(weight=weights)

    else:
        print("\n Using standard CrossEntropyLoss")
        return torch.nn.CrossEntropyLoss()