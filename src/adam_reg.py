import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
import cc3d
import struct
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch.nn as nn
import torch.optim as optim
device = 'cuda'

from kpts_util import *
#load predicted supervoxels from 3D DeepLab and estimate nonlinear displacements using Adam

#shape
H = 192; W = 160; D = 256

#Initial Supervoxels with baseline transformations (targets during training)
#see github provided download
slic = torch.load('abdomenIsoCrop38_supervoxel16_warped.pth')[27].long()
print(slic.max(),slic.shape)
#IsoCrop (no pre-alignment) MobileNet+ASPP Prediction
slic_pred = torch.load('abdomen38_mobile_superisoplus_val2.pth').long()
print(slic_pred.max(),slic_pred.shape)

#load reference image (dataset available with link in github)
img = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/img/img0038.nii.gz').get_fdata()).float()/500
seg = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/label/label0038.nii.gz').get_fdata())
img = img.cuda()

seg38 = seg.cpu().clone()*1+0
sdt = edt(seg38==0)

#example code to read in deeds displacements 
with open('baseline/F32_38_displacements.dat', 'rb') as content_file:
        content = content_file.read()
grid_space = int((torch.pow(torch.Tensor([H*W*D])/(len(content)/12),0.334)))
with torch.no_grad():
    disp_field = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(1,3,D//grid_space,W//grid_space,H//grid_space).cuda().permute(0,1,4,3,2).float()
    disp_field = F.interpolate(disp_field,size=(H,W,D),mode='trilinear',align_corners=None).permute(0,2,3,4,1)[:,:,:,:,torch.Tensor([2,0,1]).long()].flip(4)
    disp_torch = disp_field.flip(4)/torch.Tensor([256-1,160-1,192-1]).cuda().view(1,1,1,1,3)*2
identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D))
identity2 = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H//2,W//2,D//2))


#utility functions for evaluation and data loading
def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0).to(outputs.device)
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
        seg_all[i] = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/label/label00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())

        img_all[i,0] = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/img/img00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())
        #fields are not necessary for validation
        #with open('AbdomenIsoCrop/baseline/F'+str(int(cases_val[i])).zfill(2)+'_38_displacements.dat', 'rb') as content_file:
        #        content = content_file.read()
        #grid_space = int((torch.pow(torch.Tensor([H*W*D])/(len(content)/12),0.334)))
        #with torch.no_grad():
        #    disp_field = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(1,3,D//grid_space,W//grid_space,H//grid_space).cuda().permute(0,1,4,3,2).float()
        #    disp_field = F.interpolate(disp_field,size=(H,W,D),mode='trilinear',align_corners=None).permute(0,2,3,4,1)[:,:,:,:,torch.Tensor([2,0,1]).long()].flip(4)
        #    disp_torch = disp_field.flip(4)/torch.Tensor([256-1,160-1,192-1]).cuda().view(1,1,1,1,3)*2
        #fields_pairs[i] = disp_torch[:,::2,::2,::2,:].cpu()

    return img_all,seg_all,fields_pairs
#some further helper functions to interpolate sparse displacements to a dense field using TPS


#fields_pairs = F.interpolate(disp_torch.permute(0,4,1,2,3),scale_factor=0.5,mode='trilinear').permute(0,2,3,4,1)#[:,::2,::2,::2,:].cpu()

## MAIN METHOD ##

#load data
img_val,seg_val,fields_val = load_val_pairs()
img_val = img_val/500

#define Adam-Reg
lr = 0.05
alpha = 0.015#2

k = 8
num_iter = 100
N = 2048
def adam_optim(kpts_fixed, feat_kpts_fixed, feat_moving,alpha):
    class Flow(nn.Module):
        def __init__(self):
            super(Flow, self).__init__()
            self.flow = nn.Parameter(torch.zeros(kpts_fixed.shape))

        def forward(self):
            return self.flow
        
    net = Flow().to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    weight = knn_graph(kpts_fixed, k)[2]
    
    for iter in range(num_iter):
        optimizer.zero_grad()
 
        flow = net()
    
        kpts_moving = kpts_fixed + flow
        feat_kpts_moving = F.grid_sample(feat_moving, kpts_moving.view(1, 1, 1, -1, 3), mode='bilinear').view(1, -1, N).permute(0, 2, 1)
    
        data_loss = F.mse_loss(feat_kpts_moving, feat_kpts_fixed)
        reg_loss = (pdist(flow)*weight).sum()/(kpts_fixed.shape[1])
        loss = data_loss + alpha*reg_loss

        loss.backward()
        optimizer.step()
        #if(iter%20==19):
        #    print(iter)
        
    return flow.detach()
  
#convert argmax prediction to one_hot (with "compressed channels")
kpts_fixed = random_kpts(slic[0:1].unsqueeze(1)>0, 2, num_points=2048)
print(kpts_fixed.shape)
to32 = torch.zeros(16,128).long()
slic32 = torch.zeros_like(slic[:16])
slic_warp32 = torch.zeros_like(slic_warp[:16])


#run the registration for all 9 test cases
dice0 = torch.zeros(9,13)

dice_reg = torch.zeros(1,9,13)
alphas_ = torch.Tensor([0.15])
large = torch.Tensor([0,1,2,5]).long()
    
for ix in range(9):
    print('case',ix)
    with torch.no_grad():
          #convert argmax prediction to one_hot (with "compressed channels")

        for i in range(16):
            for j in range(8):
                #print(to32[i,j*32:(j+1)*32].shape,torch.randperm(32).shape)
                to32[i,j*16:(j+1)*16] = torch.randperm(16)
            slic32[i] = to32[i][slic[i]]
            slic_warp32[i] = to32[i][slic_pred[ix][i]]

        with torch.cuda.amp.autocast():

            slic32_ = F.avg_pool3d(F.avg_pool3d(F.one_hot(slic32[0:16].cuda(),16).permute(0,4,1,2,3).float(),3,stride=2,padding=1),5,stride=1,padding=2)#.cuda()

            feat_kpts_fixed = F.grid_sample(slic32_.float().reshape(1,-1,96//2,80//2,128//2), kpts_fixed.cuda().view(1, 1, 1, -1, 3), mode='bilinear').view(1, -1, N).permute(0, 2, 1)
            del slic32_
            slic_warp32_ = F.avg_pool3d(F.avg_pool3d(F.one_hot(slic_warp32[0:16].cuda(),16).permute(0,4,1,2,3).float(),3,stride=2,padding=1),5,stride=1,padding=2)#.cuda()


    seg40 = seg_val[ix]
    dice0[ix] = dice_coeff(seg40.cuda().contiguous(),seg38.cuda().contiguous(),14).cpu()
    for k in range(1):
        flow = adam_optim(kpts_fixed.cuda(), feat_kpts_fixed.cuda(), slic_warp32_.cuda().view(1,-1,96//2,80//2,128//2),alphas_[k])

        dense_flow = thin_plate_dense(kpts_fixed.cuda(), flow.cuda(), (192, 160, 256), 5, 0.001)


        seg_moving_warped = F.grid_sample(seg40.view(1,1,192,160,256).float().cuda(), F.affine_grid(torch.eye(3,4,device='cuda').unsqueeze(0), (1,1,H,W,D))\
                                      + dense_flow.cuda(), mode='nearest')#.to(device)


        dice_reg[k,ix] = dice_coeff(seg_moving_warped.contiguous(),seg38.cuda().contiguous(),14).cpu()
print('before',dice0.mean(),dice0[:,large].mean(),dice0[:,large].mean(1))#,'\n',d0)
print('slicreg',dice_reg.mean(2).mean(1),dice_reg[:,:,large].mean(2).mean(1),'\n',dice_reg[:,:,large].mean(2))#,'\n',d2)




