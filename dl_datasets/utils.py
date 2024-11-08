from torch.utils.data import default_collate

def image_collate_fn(batch):
    batch = default_collate(batch)
    return batch['jpg'], batch['cls']

def process_data(sample, transform=None):
    if transform is not None:
        sample["jpg"] = [transform(img) for img in sample["jpg"]]
    return sample