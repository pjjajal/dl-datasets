from datasets import load_dataset
from functools import partial

TOTAL_SAMPLES = 13_151_276
TOTAL_CLASSES = 19_167
URL = "timm/imagenet-w21-wds"
KEYS = {"image": "jpg", "label": "cls"}


def process_data(sample, transform=None):
    # sample["cls"] = [label for label in sample["cls"]] # subtract 1 to make it 0-indexed
    if transform is not None:
        sample["jpg"] = [transform(img) for img in sample["jpg"]]
    return sample

def imagenetw21_train(
    transform=None,
    seed=245,
    shuffle_buffer_size=1_000,
    streaming=True,
    num_proc=1,
):
    dataset = load_dataset(URL, split="train", streaming=streaming, num_proc=num_proc)
    # dataset = dataset.map(
    #     map_data,
    #     batched=True,
    #     remove_columns=dataset.column_names,
    #     num_proc=num_proc
    # )
    dataset = dataset.select_columns(['jpg', 'cls']).with_format("torch")
    dataset.set_transform(partial(process_data, transform=transform))
    return dataset
