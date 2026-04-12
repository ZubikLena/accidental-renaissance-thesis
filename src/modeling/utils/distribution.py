from collections import Counter

def get_loader_distribution(loader, max_batches=None):
    counter = Counter()

    for i, (_, labels) in enumerate(loader):
        labels = labels.numpy()

        for l in labels:
            counter[int(l)] += 1

        if max_batches and i >= max_batches:
            break
    return counter

def print_distribution(dist, label_map, task):

    if task == "binary":
        print(f"negative (0): {dist.get(0, 0)}")
        print(f"positive (1): {dist.get(1, 0)}")

    else:
        reverse_map = {v: k for k, v in label_map.items()}
        for label, count in dist.items():
            print(f"{reverse_map[label]}: {count}")