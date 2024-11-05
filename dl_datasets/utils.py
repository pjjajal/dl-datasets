from torch.utils.data import default_collate

def image_collate_fn(batch):
    batch = default_collate(batch)
    return batch['image'], batch['label']