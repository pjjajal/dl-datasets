from typing import Sequence, Literal

import torch
import torchvision.transforms.v2 as tvt


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


REGISTRY = dict()


def register_transform(func):
    REGISTRY[func.__name__] = func
    return func


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)


@register_transform
def imagenet_randaug_train(args):
    return tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.RandomResizedCrop(args.image_size),
            tvt.RandomHorizontalFlip(),
            tvt.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude),
            make_normalize_transform(),
        ]
    )


@register_transform
def imagenet_train(args):
    return tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.RandomResizedCrop(args.image_size),
            tvt.RandomHorizontalFlip(),
            make_normalize_transform(),
        ]
    )


@register_transform
def imagenet_val(args):
    return tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Resize(256, interpolation=tvt.InterpolationMode.BICUBIC),
            tvt.CenterCrop(args.image_size),
            make_normalize_transform(),
        ]
    )
