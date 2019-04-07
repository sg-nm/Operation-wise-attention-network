
# we borrowed a part of the following code:
# https://github.com/yuke93/RL-Restore

import numpy as np
import os
import h5py
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
import cv2


def load_imgs(list_in, list_gt, size = 63):
    assert len(list_in) == len(list_gt)
    img_num = len(list_in)
    imgs_in = np.zeros([img_num, size, size, 3])
    imgs_gt = np.zeros([img_num, size, size, 3])
    for k in range(img_num):
        imgs_in[k, ...] = cv2.imread(list_in[k]) / 255.
        imgs_gt[k, ...] = cv2.imread(list_gt[k]) / 255.
    return imgs_in, imgs_gt

def data_reformat(data):
    """RGB <--> BGR, swap H and W"""
    assert data.ndim == 4
    out = data[:, :, :, ::-1] - np.zeros_like(data)
    out = np.swapaxes(out, 1, 2)
    out = out.astype(np.float32)
    return out

def get_dataset_deform(train_root,val_root,test_root,is_train):
    dataset = DeformedData(
        train_root=train_root,
        val_root=val_root,
        test_root=test_root,
        is_train=is_train,
        transform=transforms.Compose([transforms.ToTensor()]),
        target_transform=transforms.Compose([transforms.ToTensor()])
    )
    return dataset

class DeformedData(data.Dataset):
    def __init__(self, train_root, val_root, test_root, is_train=0, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        self.train_dir = train_root
        self.val_dir = val_root
        self.test_dir = test_root

        if self.is_train == 0:
            # training data
            self.train_list = [self.train_dir + file for file in os.listdir(self.train_dir) if file.endswith('.h5')]
            self.train_cur = 0
            self.train_max = len(self.train_list)
            f = h5py.File(self.train_list[self.train_cur], 'r')
            self.data = f['data'].value
            self.label = f['label'].value
            f.close()
            self.data_index = 0
            self.data_len = len(self.data)
            print('training images:', self.data_len)
        elif self.is_train == 1:
            # validation data
            f = h5py.File(self.val_dir + os.listdir(self.val_dir)[0], 'r')
            self.data = f['data'].value
            self.label = f['label'].value
            f.close()
            self.data_index = 0
            self.data_len = len(self.data)
        elif self.is_train == 2:
            # # test data
            self.test_in = self.test_dir + 'moderate' + '_in/'
            self.test_gt = self.test_dir + 'moderate' + '_gt/'
            list_in = [self.test_in + name for name in os.listdir(self.test_in)]
            list_in.sort()
            list_gt = [self.test_gt + name for name in os.listdir(self.test_gt)]
            list_gt.sort()
            self.name_list = [os.path.splitext(os.path.basename(file))[0] for file in list_in]
            self.data_all, self.label_all = load_imgs(list_in, list_gt)
            self.test_total = len(list_in)
            self.test_cur = 0
            # data reformat, because the data for tools training are in a different format
            self.data = data_reformat(self.data_all)
            self.label = data_reformat(self.label_all)
            self.data_index = 0
            self.data_len = len(self.data)
        else:
            print("not implement yet")
            sys.exit()


    def __getitem__(self, index):
        img = self.data[index]
        img_gt = self.label[index]

        # transforms (numpy -> Tensor)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_gt = self.target_transform(img_gt)
        return img, img_gt

    def __len__(self):
        return self.data_len
