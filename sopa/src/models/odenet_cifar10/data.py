import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def get_cifar10_train_val_loaders(data_aug=False,
                                  batch_size=128,
                                  val_perc=0.1,
                                  data_root=None,
                                  num_workers=1,
                                  pin_memory=True,
                                  shuffle=True,
                                  random_seed=None,
                                  download=False):
    '''  Returns iterators through train/val CIFAR10 datasets.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    :param data_aug: bool
        Whether to apply the data augmentation scheme. Only applied on the train split.
    :param batch_size: int
        How many samples per batch to load.
    :param val_perc: float
        Percentage split of the training set used for the validation set. Should be a float in the range [0, 1].
    :param data_root: str
        Path to the directory with the dataset.
    :param num_workers: int
        Number of subprocesses to use when loading the dataset.
    :param pin_memory: bool
        Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :param shuffle: bool
        Whether to shuffle the train/validation indices
    :param random_seed: int
        Fix seed for reproducibility.
    :return:
    '''

    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=download, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=data_root, train=True, download=download, transform=transform_test)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_perc * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              drop_last=True, pin_memory=pin_memory,)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                              drop_last=True, pin_memory=pin_memory,)

    return train_loader, val_loader


def get_cifar10_test_loader(batch_size=128, data_root=None, num_workers=1, pin_memory=True, shuffle=False, download=False):
    ''' Returns iterator through CIFAR10 test dataset
    
    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    :param batch_size: int
        How many samples per batch to load.
    :param data_root: str
        Path to the directory with the dataset.
    :param num_workers: int
        Number of subprocesses to use when loading the dataset.
    :param pin_memory: bool
        Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :param shuffle: bool
        Whether to shuffle the dataset after every epoch.
    :return: 
    '''
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             drop_last=True, pin_memory=pin_memory)
    return test_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

