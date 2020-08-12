import os
import numpy as np
import torch
from torchvision import datasets, transforms
from cub2011 import Cub2011


def get_transforms():
    train_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(size=227),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(size=227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return train_transform, val_transform

def get_inverse_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inv_normalize = transforms.Normalize(
       mean= [-m/s for m, s in zip(mean, std)],
       std= [1/s for s in std]
    )
    return inv_normalize

class ClassDisjointDataSet(torch.utils.data.Dataset):
    def __init__(self, original_train, original_val, train, split_point=0.5, n_classes=100, transform=None):
        self.original_train = original_train
        self.original_val = original_val

        split_class = round(split_point*n_classes)
        rule = (lambda x: x < split_class) if train else (lambda x: x >=split_class)
        train_filtered_idx = [i for i,x in enumerate(original_train.data.target) if rule(x)]
        val_filtered_idx = [i for i,x in enumerate(original_val.data.target) if rule(x)]
        
        train_fpaths = original_train.data.iloc[train_filtered_idx].apply(
            lambda x: os.path.join(self.original_train.root, self.original_train.base_folder, x.filepath),
            axis=1
        )
        val_fpaths = original_val.data.iloc[val_filtered_idx].apply(
            lambda x: os.path.join(self.original_val.root, self.original_val.base_folder, x.filepath),
            axis=1
        )

        self.img_paths = np.concatenate([train_fpaths, val_fpaths])
        self.data = np.concatenate([original_train.data.iloc[train_filtered_idx], original_val.data.iloc[val_filtered_idx]], axis=0)
        self.targets = np.concatenate([np.array(original_train.data.target)[train_filtered_idx], np.array(original_val.data.target)[val_filtered_idx]], axis=0)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):            
        path, target = self.img_paths[index], self.targets[index]
        img = datasets.folder.default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_cub_2011():
    original_train = Cub2011(root="CUB2011", train=True, transform=None, download=False)
    original_val = Cub2011(root="CUB2011", train=False, transform=None, download=False)
    return original_train, original_val

def get_cub_2011_class_disjoint(train_transform, val_transform):
    original_train, original_val = get_cub_2011()
    n_classes = max(max(original_train.data.target), max(original_val.data.target))
    train_dataset = ClassDisjointDataSet(original_train, original_val, True, n_classes=n_classes, transform=train_transform)
    val_dataset = ClassDisjointDataSet(original_train, original_val, False, n_classes=n_classes, transform=val_transform)
    assert set(train_dataset.targets).isdisjoint(set(val_dataset.targets))
    return train_dataset, val_dataset