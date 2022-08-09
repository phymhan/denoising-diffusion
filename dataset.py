from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from tensorfn.data import LMDBReader
from torchvision.datasets import LSUNClass
import torch
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as F
import pickle
import os
import numpy as np
import tqdm
import random
from natsort import natsorted

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        # PIL size is (width, height), torchvision crop is (height, width)!!!
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
            else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
            else np.random.randint(low=0,high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.reader = LMDBReader(path, reader="raw")

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        img_bytes = self.reader.get(
            f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
        )

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class LmdbDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.dataset = LSUNClass(root=os.path.expanduser(path))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img = self.transform(img)
        return img


def get_image_dataset(args, which_dataset='c10', data_root='./data', train=True, random_crop=False):
    # Define Image Datasets (VideoFolder will be the collection of all frames)
    CropLongEdge = RandomCropLongEdge if train else CenterCropLongEdge
    dataset = None
    if which_dataset.lower() in ['multires']:
        transform = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
        if random_crop:
            transform += [
                transforms.RandomCrop(args.crop_size)
            ]
        transform = transforms.Compose(transform)
        dataset = MultiResolutionDataset(data_root, transform, args.size)
    elif which_dataset.lower() in ['lmdb', 'lsun']:
        transform = [
            CropLongEdge(),
            transforms.Resize(args.size, Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if random_crop:
            transform += [
                transforms.RandomCrop(args.crop_size)
            ]
        transform = transforms.Compose(transform)
        dataset = LmdbDataset(
            path=data_root,
            transform=transform
        )
    elif which_dataset.lower() in ['imagefolder', 'custom']:
        transform = [
            CropLongEdge(),
            transforms.Resize(args.size, Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if random_crop:
            transform += [
                transforms.RandomCrop(args.crop_size)
            ]
        transform = transforms.Compose(transform)
        dataset = torchvision.datasets.ImageFolder(
            root=data_root,
            transform=transform
        )
    elif which_dataset.lower() in ['mnist']:
        transform = [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transform)
        dataset = torchvision.datasets.MNIST(
            root=data_root, download=True,
            train=train,
            transform=transform
        )
    elif which_dataset.lower() in ['cifar10', 'c10']:
        transform = [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transform)
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, download=True,
            train=train,
            transform=transform
        )
    elif which_dataset.lower() in ['cifar100', 'c100']:
        transform = [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transform)
        dataset = torchvision.datasets.CIFAR100(
            root=data_root, download=True,
            train=train,
            transform=transform
        )
    elif which_dataset.lower() in ['imagenet', 'ilsvrc2012']:
        # TODO: save file index, hdf5 or lmdb
        transform = [
            CropLongEdge(),
            transforms.Resize(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transform)
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_root, 'train' if train else 'valid'),
            transform=transform
        )
    elif which_dataset.lower() in ['tiny_imagenet', 'tiny']:
        transform = [
            transforms.Resize(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transform)
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_root, 'train' if train else 'test'),
            transform=transform
        )
    else:
        raise NotImplementedError
    return dataset