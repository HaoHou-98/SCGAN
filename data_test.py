import os,time,sys
from glob import glob, iglob
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL
import torch


class faces_data(data.Dataset):
    def __init__(self, data_hr, data_lr):
        self.hr_imgs = [os.path.join(d, i) for d in data_hr for i in os.listdir(d) if
                        os.path.isfile(os.path.join(d, i))]
        self.lr_imgs = [os.path.join(d, i) for d in data_lr for i in os.listdir(d) if
                        os.path.isfile(os.path.join(d, i))]
        self.lr_len = len(self.lr_imgs)
        self.lr_shuf = np.arange(self.lr_len)
        np.random.shuffle(self.lr_shuf)
        self.lr_idx = 0
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.imread(self.hr_imgs[index])
        lr = cv2.imread(self.lr_imgs[self.lr_shuf[self.lr_idx]])
        self.lr_idx += 1
        if self.lr_idx >= self.lr_len:
            self.lr_idx = 0
            np.random.shuffle(self.lr_shuf)
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["lr"] = self.preproc(lr)
        data["hr"] = self.preproc(hr)
        data["hr_down"] = nnF.avg_pool2d(data["hr"], 4, 4)
        return data

    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)

class faces_super(data.Dataset):
    def __init__(self, datasets, transform):
        assert datasets, print('no datasets specified')
        self.transform = transform
        self.img_list = []
        dataset = datasets
        if dataset == 'LS3D':
            img_path = './imgs_test/LS3D-w_balanced'
            list_name = (glob(os.path.join(img_path, "*.png")))
            list_name.sort()
            for filename in list_name:
                self.img_list.append(filename)
        if dataset == 'FFHQ':
            img_path = './imgs_test/FFHQ'
            list_name = (glob(os.path.join(img_path, "*.png")))
            list_name.sort()
            for filename in list_name:
                self.img_list.append(filename)
        if dataset == 'Widerface':
            img_path = './imgs_test/Widerface'
            list_name = (glob(os.path.join(img_path, "*.png")))
            list_name.sort()
            for filename in list_name:
                self.img_list.append(filename)
        if dataset == 'WebFace':
            img_path = './imgs_test/WebFace'
            list_name = (glob(os.path.join(img_path, "*.png")))
            list_name.sort()
            for filename in list_name:
                self.img_list.append(filename)
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        inp16 = Image.open(self.img_list[index])
        inp64 = inp16.resize((64, 64), resample=PIL.Image.BICUBIC)
        data['img64'] = self.transform(inp64)
        data['img16'] = self.transform(inp16)
        data['imgpath'] = self.img_list[index]
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        return data

def get_loader(dataname,bs =1):
    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = faces_super(dataname, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader