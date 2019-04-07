#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
from skimage.measure import compare_psnr as ski_psnr
from skimage.measure import compare_ssim as ski_ssim
import os
import logging

from model import Network
import torch.nn.functional as F
from data_load_own import get_training_set, get_test_set
from data_load_mix import get_dataset_deform
import utils
import argparse



parser = argparse.ArgumentParser(description='Operation-wise Attention Network')
parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
parser.add_argument('--mode', '-m', default='mix', help='Mode (mix / yourdata)')
args = parser.parse_args()

# load dataset
if args.mode == 'mix' or args.mode == 'yourdata':
    if args.mode == 'mix':
        num_work = 8
        train_dir = '/dataset/train/'
        val_dir = '/dataset/val/'
        test_dir = '/dataset/test/'
        test_set = get_dataset_deform(train_dir, val_dir, test_dir, 2)
        test_dataloader = DataLoader(dataset=test_set, num_workers=num_work, batch_size=1, shuffle=False, pin_memory=False)
    elif args.mode == 'yourdata':
        num_work = 8
        test_input_dir = '/dataset/yourdata_test/input/'
        test_target_dir = '/dataset/yourdata_test/target/'
        test_set = get_training_set(test_input_dir, test_target_dir, False)
        test_dataloader = DataLoader(dataset=test_set, num_workers=num_work, batch_size=1, shuffle=False, pin_memory=False)
else:
    print('\tInvalid input dataset name at CNN_train()')
    exit(1)


# model
gpuID = 0
torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
L1_loss = nn.L1Loss()
L1_loss = L1_loss.cuda(gpuID)
model = Network(16, 10, L1_loss, gpuID=gpuID)
if args.gpu_num == 1:
    ## not using dataparallel in training
    model.load_state_dict(torch.load('./Trained_model/model_best.pth'))
else:
    ## using dataparallel in training
    state_dict = torch.load('./Trained_model/model_best.pth')
    ## when you use gpuID != 0 for training, you need to specify the gpuID below.
    # state_dict = torch.load('./model_best.pth', map_location={'cuda:3':'cuda:0'})
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

model = model.cuda(gpuID)
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
print('Param:', utils.count_parameters_in_MB(model))
# for results
if not os.path.exists('./results'):
    os.makedirs('./results/Inputs')
    os.makedirs('./results/Outputs')
    os.makedirs('./results/Targets')


test_ite = 0
test_psnr = 0
test_ssim = 0
eps = 1e-10
for i, (input, target) in enumerate(test_dataloader):
    lr_patch = Variable(input, requires_grad=False).cuda(gpuID)
    hr_patch = Variable(target, requires_grad=False).cuda(gpuID)
    output = model(lr_patch)
    # save images
    vutils.save_image(output.data, './results/Outputs/%05d.png' % (int(i)), padding=0, normalize=False)
    vutils.save_image(lr_patch.data, './results/Inputs/%05d.png' % (int(i)), padding=0, normalize=False)
    vutils.save_image(hr_patch.data, './results/Targets/%05d.png' % (int(i)), padding=0, normalize=False)
    # SSIM and PSNR
    output = output.data.cpu().numpy()[0]
    output[output>1] = 1
    output[output<0] = 0
    output = output.transpose((1,2,0))
    hr_patch = hr_patch.data.cpu().numpy()[0]
    hr_patch[hr_patch>1] = 1
    hr_patch[hr_patch<0] = 0
    hr_patch = hr_patch.transpose((1,2,0))
    # SSIM
    test_ssim+= ski_ssim(output, hr_patch, data_range=1, multichannel=True)
    # PSNR
    imdf = (output - hr_patch) ** 2
    mse = np.mean(imdf) + eps
    test_psnr+= 10 * math.log10(1.0/mse)
    test_ite += 1
test_psnr /= (test_ite)
test_ssim /= (test_ite)
print('Test PSNR: {:.4f}'.format(test_psnr))
print('Test SSIM: {:.4f}'.format(test_ssim))
print('------------------------')
