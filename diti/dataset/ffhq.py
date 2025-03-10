import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import open_lmdb

class FFHQ(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config["image_size"]
        self.image_channel = self.config["image_channel"]
        self.data_path = self.config["data_path"]
        self.augmentation = self.config["augmentation"]

        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            if "ssl_aug" in self.config.keys():
                ssl_aug = self.config["ssl_aug"]
            else:
                ssl_aug = "simclr_waug"

            if ssl_aug == "simsiam":
                self.ssl_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(self.image_size, scale=(0.25, 1.0)),
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            elif ssl_aug == "simclr":
                s = 1.0
                self.ssl_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(self.image_size, scale=(0.25, 1.0)),
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            elif ssl_aug == "simclr_waug":
                s = 1.0
                self.ssl_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(self.image_size, scale=(0.25, 1.0)),
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            elif ssl_aug == "simclr_waug2":
                s = 1.0
                self.ssl_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(self.image_size, scale=(0.25, 1.0)),
                    transforms.Resize(self.image_size),
                    transforms.RandomApply([transforms.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            elif ssl_aug == "mae":
                self.ssl_transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                assert False, 'Specified SSL aug not found!'

        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.ssl_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

        image_dm = self.transform(image)
        image_ssl1 = self.ssl_transform(image)
        image_ssl2 = self.ssl_transform(image)
        gt = image_dm.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return {
            "x_0": image_dm,
            "x_ssl1": image_ssl1,
            "x_ssl2": image_ssl2,
            "gt": gt,
            "x_T": torch.randn(self.image_channel, self.image_size, self.image_size),
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        x_0 = []
        x_ssl1=[]
        x_ssl2=[]
        gts = []
        x_T =[]

        for i in range(batch_size):
            x_0.append(batch[i]["x_0"])
            x_ssl1.append(batch[i]["x_ssl1"])
            x_ssl2.append(batch[i]["x_ssl2"])
            gts.append(batch[i]["gt"])
            x_T.append(batch[i]["x_T"])

        x_0 = torch.stack(x_0, dim=0)
        x_ssl1 = torch.stack(x_ssl1, dim=0)
        x_ssl2 = torch.stack(x_ssl2, dim=0)
        x_T = torch.stack(x_T, dim=0)

        return {
            "net_input": {
                "x_0": x_0,
                "x_ssl1": x_ssl1,
                "x_ssl2": x_ssl2,
                "x_T": x_T,
            },
            "gts": np.asarray(gts),
        }
