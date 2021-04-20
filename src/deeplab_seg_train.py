import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
#import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch import Tensor
import time       
import nibabel as nib
import numpy as np
import struct
from torch.utils.checkpoint import checkpoint
from typing import Callable, Any, Optional, List

import os
H = 192
W = 160
D = 192#256
def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))
    
#train 3D DeepLab to predict supervoxels
from models import *

#evaluate segmentation accuracy
def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice

def load_val_pairs():
    cases_val = torch.Tensor([1,4,7,10,23,26,29,32,35,38])


    fields_pairs = torch.zeros(10,192//2,160//2,256//2,3)
        

    seg_all = torch.zeros(10,192,160,256).long()
    img_all = torch.zeros(10,1,H,W,D)

    for i in range(10):
        seg_all[i] = torch.from_numpy(nib.load('/share/data_hoots1/heinrich/AbdomenIsoCrop/Training/label/label00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())

        img_all[i,0] = torch.from_numpy(nib.load('/share/data_hoots1/heinrich/AbdomenIsoCrop/Training/img/img00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())
        with open('/share/data_hoots1/heinrich/AbdomenIsoCrop/baseline/F'+str(int(cases_val[i])).zfill(2)+'_38_displacements.dat', 'rb') as content_file:
                content = content_file.read()
        grid_space = int((torch.pow(torch.Tensor([H*W*D])/(len(content)/12),0.334)))
        with torch.no_grad():
            disp_field = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(1,3,D//grid_space,W//grid_space,H//grid_space).cuda().permute(0,1,4,3,2).float()
            disp_field = F.interpolate(disp_field,size=(H,W,D),mode='trilinear',align_corners=None).permute(0,2,3,4,1)[:,:,:,:,torch.Tensor([2,0,1]).long()].flip(4)
            disp_torch = disp_field.flip(4)/torch.Tensor([256-1,160-1,192-1]).cuda().view(1,1,1,1,3)*2
        fields_pairs[i] = disp_torch[:,::2,::2,::2,:].cpu()

    return img_all,seg_all,fields_pairs
    
def load_train_pairs():
    cases_val = torch.Tensor([2,3,5,6,8,9,21,22,24,25,27,28,30,31,33,34,36,37,39,40])#1,4,7,10,23,26,29,32,35,38])

    fields_pairs = torch.zeros(20,192//2,160//2,256//2,3)
   
    seg_all = torch.zeros(20,192,160,256).long()
    img_all = torch.zeros(20,1,H,W,D)

    for i in range(20):
        seg_all[i] = torch.from_numpy(nib.load('/share/data_hoots1/heinrich/AbdomenIsoCrop/Training/label/label00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())
        img_all[i,0] = torch.from_numpy(nib.load('/share/data_hoots1/heinrich/AbdomenIsoCrop/Training/img/img00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())
        with open('/share/data_hoots1/heinrich/AbdomenIsoCrop/baseline/F'+str(int(cases_val[i])).zfill(2)+'_38_displacements.dat', 'rb') as content_file:
                content = content_file.read()
        grid_space = int((torch.pow(torch.Tensor([H*W*D])/(len(content)/12),0.334)))
        with torch.no_grad():
            disp_field = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(1,3,D//grid_space,W//grid_space,H//grid_space).cuda().permute(0,1,4,3,2).float()
            disp_field = F.interpolate(disp_field,size=(H,W,D),mode='trilinear',align_corners=None).permute(0,2,3,4,1)[:,:,:,:,torch.Tensor([2,0,1]).long()].flip(4)
            disp_torch = disp_field.flip(4)/torch.Tensor([256-1,160-1,192-1]).cuda().view(1,1,1,1,3)*2
        fields_pairs[i] = disp_torch[:,::2,::2,::2,:].cpu()


    return img_all,seg_all,fields_pairs
