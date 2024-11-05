from datasets import load_dataset
from functools import partial

TOTAL_SAMPLES = 13_151_276
TOTAL_CLASSES = 19_167
URL = "timm/imagenet-w21-wds"
KEYS = {"image": "jpg", "label": "cls"}


def process_data(sample, transform=None):
    sample["label"] = [label - 1 for label in sample[KEYS["label"]]] # subtract 1 to make it 0-indexed
    if transform is not None:
        sample["image"] = [transform(img) for img in sample[KEYS["image"]]]
    return sample


def imagenetw21_train(
    transform=None,
    seed=245,
    shuffle_buffer_size=1_000,
    streaming=True,
):
    dataset = load_dataset(URL, split="train", streaming=streaming)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size).map(
        partial(process_data, transform=transform),
        batched=True,
        remove_columns=dataset.column_names,
    ).with_format("torch")

    return dataset
