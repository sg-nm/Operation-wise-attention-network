from torchvision.transforms import Compose, CenterCrop, ToTensor
import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image
import random
from random import randrange


def input_transform():
    return Compose([
        #CenterCrop(crop_size),
        #Resize(crop_size // upscale_factor),
        ToTensor(),
    ])

def target_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(input, target, data_augmentation):
    return DatasetFromFolder(target, input, data_augmentation,
                             input_transform=input_transform(),
                             target_transform=target_transform())


def get_test_set(lr_dir, hr_dir, upscale_factor, patch_size, data_augmentation=False):
    return DatasetFromFolderTest(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation=False,
                             input_transform=input_transform(),
                             target_transform=target_transform())

def get_eval_set(lr_dir):
    return DatasetFromFolderEval(lr_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def get_patch(img_in, img_tar,patch_size, scale, ix=-1, iy=-1):
    (c, ih, iw) = img_in.shape
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar[:, ty:ty + tp, tx:tx + tp]
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch

def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = torch.from_numpy(img_in.numpy()[:, :, ::-1].copy())
        img_tar = torch.from_numpy(img_tar.numpy()[:, :, ::-1].copy())
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = torch.from_numpy(img_in.numpy()[:, ::-1, :].copy())
            img_tar = torch.from_numpy(img_tar.numpy()[:, ::-1, :].copy())
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = torch.FloatTensor(np.transpose(img_in.numpy(),(0,2,1)))
            img_tar = torch.FloatTensor(np.transpose(img_tar.numpy(),(0,2,1)))
            info_aug['trans'] = True

    return img_in, img_tar, info_aug
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, hr_dir, lr_dir, data_augmentation, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames_inp = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.image_filenames_trg = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.image_filenames_trg[index])
        input  = load_img(self.image_filenames_inp[index])
               
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        
        if self.data_augmentation:
            input, target, _ = augment(input, target)
        return input, target

    def __len__(self):
        return len(self.image_filenames_inp)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation, input_transform=None, target_transform=None):
        super(DatasetFromFolderTest, self).__init__()
        self.image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        tmp = os.path.splitext(file)[0]
        tmp = tmp.split("_")[0]
        input = load_img(os.path.join(self.lr_dir, tmp+'_LRBI_'+'x'+str(self.upscale_factor)+'.png'))
               
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        
        if self.data_augmentation:
            input, target, _ = augment(input, target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir,input_transform=None, target_transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        if self.input_transform:
            input = self.input_transform(input)
        return input, file
      
    def __len__(self):
        return len(self.image_filenames)
