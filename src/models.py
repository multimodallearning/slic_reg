import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from torch import nn
from torch.nn import functional as F
#import matplotlib.pyplot as plt


from torch import Tensor
import time       
import numpy as np
import struct
from torch.utils.checkpoint import checkpoint
    
#from .utils import load_state_dict_from_url
from typing import Callable, Any, Optional, List

import os
H = 192
W = 160
D = 192#256
#Atrous Spatial Pyramid Pooling (Segmentation Network)
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            #nn.AdaptiveAvgPool2d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-3:]
        x = F.adaptive_avg_pool3d(x,(1))
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='nearest')#, align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
class OBELISK(nn.Module):
    def __init__(self):

        super(OBELISK, self).__init__()
        channels = 64
        self.ogrid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H//8,W//8,D//8)).view(1,1,-1,1,3)

        self.offsets = nn.Parameter(torch.randn(1,channels,1,1,3)*0.075)
        self.layer1 = nn.Conv3d(channels*9, channels*2, 1, bias=False, groups=1)
        self.batch1 = nn.BatchNorm3d(channels*2)
        self.layer2 = nn.Conv3d(channels*2, channels*2, 3, bias=False, padding=1)
        self.batch2 = nn.BatchNorm3d(channels*2)
        #self.layer3 = nn.Conv3d(channels*2, channels*1, 1)


    def forward(self, feat):
        B = feat.size(0)
        #print('feat',feat.size())
        grid = self.ogrid_xyz.to(feat.device).to(feat.dtype)
        sampled = F.grid_sample(feat.view(B*8,-1,H//8,W//8,D//8), grid + self.offsets.to(feat.dtype).repeat(B,1,1,1,1).view(B*8,-1,1,1,3)).view(B,-1,H//8,W//8,D//8).to(feat.dtype)
        #print('sampled',sampled.shape)
        x = F.relu(self.batch1(self.layer1(torch.cat((sampled,feat),1))))
        x = F.relu(self.batch2(self.layer2(x)))
        return x
#Mobile-Net with depth-separable convolutions and residual connections
class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
def create_model():
#    in_channels = torch.Tensor([1,16,24,24,32,32,32,64]).long()
    in_channels = torch.Tensor([1,24,24,32,48,48,48,64]).long()

    in_channels[0] = 1
    mid_channels = torch.Tensor([64,128,192,192,256,256,256,384]).long()
    out_channels = torch.Tensor([24,24,32,48,48,48,64,64]).long()
    mid_stride = torch.Tensor([1,1,1,2,1,1,1,1])
    net = []
    net.append(nn.Identity())
    for i in range(8):
        inc = int(in_channels[i]); midc = int(mid_channels[i]); outc = int(out_channels[i]); strd = int(mid_stride[i])
        layer = nn.Sequential(nn.Conv3d(inc,midc,1,bias=False),nn.BatchNorm3d(midc),nn.ReLU6(True),\
                        nn.Conv3d(midc,midc,3,stride=strd,padding=1,bias=False,groups=midc),nn.BatchNorm3d(midc),nn.ReLU6(True),\
                                       nn.Conv3d(midc,outc,1,bias=False),nn.BatchNorm3d(outc))
        if(i==0):
            layer[0] = nn.Conv3d(inc,midc,3,padding=1,stride=2,bias=False)
        if((inc==outc)&(strd==1)):
            net.append(ResBlock(layer))
        else:
            net.append(layer)

    backbone = nn.Sequential(*net)

    count = 0
    # weight initialization
    for m in backbone.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            count += 1
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    print('#CNN layer',count)

    #complete model: MobileNet + ASPP + head (with a single skip connection)
    #newer model (one more stride, no groups in head)
    aspp = ASPP(64,(2,4,8,16),128)#ASPP(64,(1,),128)#
    num_classes = 128#
    #head with 16 groups for 16 layers of supervoxels
    head = nn.Sequential(nn.Conv3d((128+24), 32*16, 1, padding=0,groups=1, bias=False),nn.BatchNorm3d(32*16),nn.ReLU(),\
                         nn.Conv3d(32*16, 32*16, 3, groups=16,padding=1, bias=False),nn.BatchNorm3d(32*16),nn.ReLU(),\
                         nn.Conv3d(32*16, num_classes*16, 1,groups=16))
    return backbone,aspp,head
    
