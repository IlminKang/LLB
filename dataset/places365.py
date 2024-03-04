import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import os
import pandas as pd


class Places365(Dataset):
    def __init__(self, root_dir, transform, split='train'):
        self.root_dir = root_dir

        self.root_dir = f'{root_dir}/{split}'
        self.data = datasets.ImageFolder(self.root_dir, transform=transform)


    def __len__(self):
        return len(self.data)

    def class_to_idx(self):
        return self.data.class_to_idx

    def classes(self):
        return self.data.classes

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # cls = self.indicator['label'][idx]

        return image, label
