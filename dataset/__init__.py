import os
import torch
import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import InterpolationMode


def get_dataset(args, split='train', eval=True):
    '''
    :param args: hyper-parameters
    :param split: split
    :return: dataset instance
    '''
    if split not in ['train', 'val', 'test']:
        raise TypeError(f'Invalid split type! {split}')

    if args.dataset.lower() == 'imagenet':
        from dataset.imagenet import ImageNet as Dataset_

        if 'augreg' in args.method:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        if split == 'val':
            return None
        if 'swag' in args.method:
            if split == 'test':
                split = 'val'
                resize = 384
                size = 384
            if split == 'train':
                resize = 384
                size = 384
        else:
            if split == 'test':
                split = 'val'
                size = 224
                resize = 256
            if split == 'train':
                size = 224
                resize = 256


    elif args.dataset.lower() == 'places365':
        from dataset.places365 import Places365 as Dataset_

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if split == 'val':
            return None
        if 'swag' in args.method:
            if split == 'test':
                split = 'val'
                resize = 384
                size = 384
            if split == 'train':
                resize = 384
                size = 384
        else:
            if split == 'test':
                split = 'val'
                size = 224
                resize = (256, 256)
            if split == 'train':
                size = 224
                resize = 256

    elif args.dataset.lower() == 'inaturalist2018':
        from dataset.inaturalist import iNaturalist2018 as Dataset_

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if split == 'val':
            return None

        if 'swag' in args.method:
            if split == 'test':
                split = 'val'
                resize = 384
                size = 384
            if split == 'train':
                resize = 384
                size = 384
        else:
            if split == 'test':
                split = 'val'
                size = 224
                resize = 256
            if split == 'train':
                size = 224
                resize = 256

    else:
        raise TypeError(f'Invalid dataset! : {args.dataset}')

    root_dir = args.data_dir

    if split == 'train':

        transform = transforms.Compose([transforms.Resize(resize),
                                        transforms.CenterCrop(size),
                                        transforms.RandAugment(num_ops=3, magnitude=9),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                        ])

    else:
        transform = transforms.Compose([
            # transforms.Resize(resize),
            transforms.Resize(
                resize,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    dataset = Dataset_(root_dir=root_dir,
                       transform=transform,
                       split=split)

    print(f'\t|-{args.dataset} {split}: {len(dataset)} - {size} - M{mean} - Std{std} ')
    return dataset
