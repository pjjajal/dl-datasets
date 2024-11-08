from datasets import load_dataset
from functools import partial
from .utils import process_data

TOTAL_SAMPLES = 13_151_276
TOTAL_VAL_SAMPLES = None  # no validation set
TOTAL_CLASSES = 19_167
URL = "timm/imagenet-w21-wds"
KEYS = {"image": "jpg", "label": "cls"}


def imagenetw21_train(transform=None, num_proc=1):
    dataset = load_dataset(URL, split="train", num_proc=num_proc)
    dataset = dataset.select_columns(["jpg", "cls"]).with_format("torch")
    dataset.set_transform(partial(process_data, transform=transform))
    return dataset
