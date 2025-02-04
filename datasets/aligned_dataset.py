import os.path
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from .image_folder import make_dataset
from PIL import Image

import torchvision
import blobfile as bf
import torchvision.transforms.functional as F
from glob import glob

from utils.utils import open_lmdb
from io import BytesIO

import pandas as pd

def get_params( size,  resize_size,  crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}
 
class CropCelebA64(object):
    def __call__(self, img):
        new_img = F.crop(img, 57, 25, 128, 128)
        return new_img

def get_transform(params,  resize_size,  crop_size, method=Image.BICUBIC,  flip=True, crop = True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, crop_size, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        return torch.nn.functional.interpolate(img.unsqueeze(0), size=(target_width, target_width), mode='bicubic', align_corners=False).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)

def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)


class FFHQ(torch.utils.data.Dataset):
    def __init__(self, dataroot, train=True,  img_size=128, augmentation=True):
        super().__init__()
        self.image_size = img_size
        self.image_channel = 3
        self.data_path = dataroot
        self.augmentation = augmentation

        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return 70000

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)
        key = f'256-{str(index).zfill(5)}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return image, gt, index


    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        gt=[]

        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
        }

class CELEBAHQ(torch.utils.data.Dataset):
    #def __init__(self, config):
    def __init__(self, dataroot, train=True,  img_size=128, augmentation=True):
        super().__init__()
        self.image_size = img_size
        self.image_channel = 3
        self.data_path = dataroot
        self.augmentation = augmentation

        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return 30000

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)
        key = f'128-{str(index).zfill(5)}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return image, gt, index


    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        gt=[]

        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
        }

class CELEBA64(torch.utils.data.Dataset):
    def __init__(self, dataroot, split="train", img_size=64, augmentation=True):
        super().__init__()
        self.image_size = img_size
        self.image_channel = 3
        self.data_path = dataroot
        self.augmentation = augmentation
        self.split = split

        if self.augmentation:
            self.transform = transforms.Compose([
                CropCelebA64(),
                transforms.Resize((self.image_size,self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                CropCelebA64(),
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
            ])

    # train: 0~162,769  162,770
    # valid: 162,770~182,636   19,867
    # test: 182,637~202,599   19,963
    def __len__(self):
        if self.split == "train":
            return 162770
        if self.split == "valid":
            return 19867
        if self.split == "test":
            return 19963
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        if self.split == "train":
            offset_index = index
        elif self.split == "valid":
            offset_index = 162770 + index
        elif self.split == "test":
            offset_index = 162770 + 19867 + index
        else:
            raise NotImplementedError

        key = f'None-{str(offset_index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return image, gt, index

class HORSE(torch.utils.data.Dataset):
    def __init__(self, dataroot, train=True,  img_size=128, augmentation=True):
        super().__init__()
        self.image_size = img_size
        self.image_channel = 3
        self.data_path = dataroot
        self.augmentation = augmentation

        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return 2000340

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return image, gt, index

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        gt=[]

        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
        }

class BEDROOM(torch.utils.data.Dataset):
    def __init__(self, dataroot, train=True,  img_size=128, augmentation=True):
        super().__init__()
        self.image_size = img_size
        self.image_channel = 3
        self.data_path = dataroot
        self.augmentation = augmentation

        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return 3033042

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return image, gt, index

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        gt =[]
        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
        }


    

