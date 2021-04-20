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
        seg_all[i] = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/label/label00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())

        img_all[i,0] = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/img/img00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())
        #with open('AbdomenIsoCrop/baseline/F'+str(int(cases_val[i])).zfill(2)+'_38_displacements.dat', 'rb') as content_file:
        #        content = content_file.read()
        #grid_space = int((torch.pow(torch.Tensor([H*W*D])/(len(content)/12),0.334)))
        #with torch.no_grad():
        #    disp_field = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(1,3,D//grid_space,W//grid_space,H//grid_space).cuda().permute(0,1,4,3,2).float()
        #    disp_field = F.interpolate(disp_field,size=(H,W,D),mode='trilinear',align_corners=None).permute(0,2,3,4,1)[:,:,:,:,torch.Tensor([2,0,1]).long()].flip(4)
        #    disp_torch = disp_field.flip(4)/torch.Tensor([256-1,160-1,192-1]).cuda().view(1,1,1,1,3)*2
        #fields_pairs[i] = disp_torch[:,::2,::2,::2,:].cpu()

    return img_all,seg_all,fields_pairs
    
def load_train_pairs():
    cases_val = torch.Tensor([2,3,5,6,8,9,21,22,24,25,27,28,30,31,33,34,36,37,39,40])#1,4,7,10,23,26,29,32,35,38])

    fields_pairs = torch.zeros(20,192//2,160//2,256//2,3)
   
    seg_all = torch.zeros(20,192,160,256).long()
    img_all = torch.zeros(20,1,H,W,D)

    for i in range(20):
        seg_all[i] = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/label/label00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())
        img_all[i,0] = torch.from_numpy(nib.load('AbdomenIsoCrop/Training/img/img00'+str(int(cases_val[i])).zfill(2)+'.nii.gz').get_fdata())
        #with open('AbdomenIsoCrop/baseline/F'+str(int(cases_val[i])).zfill(2)+'_38_displacements.dat', 'rb') as content_file:
        #        content = content_file.read()
        #grid_space = int((torch.pow(torch.Tensor([H*W*D])/(len(content)/12),0.334)))
        #with torch.no_grad():
        #    disp_field = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(1,3,D//grid_space,W//grid_space,H//grid_space).cuda().permute(0,1,4,3,2).float()
        #    disp_field = F.interpolate(disp_field,size=(H,W,D),mode='trilinear',align_corners=None).permute(0,2,3,4,1)[:,:,:,:,torch.Tensor([2,0,1]).long()].flip(4)
        #    disp_torch = disp_field.flip(4)/torch.Tensor([256-1,160-1,192-1]).cuda().view(1,1,1,1,3)*2
        #fields_pairs[i] = disp_torch[:,::2,::2,::2,:].cpu()


    return img_all,seg_all,fields_pairs

import numpy as np
def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
H = 192
W = 160
D = 256
gpu_usage()

backbone,aspp,head = create_model()
print('#param(backbone)',countParameters(backbone),'#param(aspp)',countParameters(aspp),'#param(head)',countParameters(head))

img_val,seg_val,fields_val = load_val_pairs()
img_val = img_val/500
#seg_val = seg_val.cuda()
#fields_pairs,seg_all,idx_pair = load_pairs()
#img_val = F.interpolate(img_val,scale_factor=0.5,mode='trilinear').cuda()/500
print('validation data loaded')

img_train,seg_train,fields_train = load_train_pairs()
img_train = img_train/500
#fields_pairs,seg_all,idx_pair = load_pairs()
#img_val = F.interpolate(img_val,scale_factor=0.5,mode='trilinear').cuda()/500
print('training data loaded')

#prepared target SLIC supervoxels (available for download)
slic = torch.load('abdomenIsoCrop38_supervoxel16_warped.pth')
slic_train = slic[torch.Tensor([1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]).long()]
slic_val = slic[torch.Tensor([0,3,6,9,12,15,18,21,24,27]).long()]
del slic

#idx = torch.arange(4)
#with torch.no_grad():
#    identity2 = F.affine_grid(torch.eye(3,4).unsqueeze(0)+torch.randn(4,3,4)*.0,(4,1,96,80,128)).cuda()
#    slic_warp = F.grid_sample(slic.float().unsqueeze(0).cuda().repeat(4,1,1,1,1),fields_val[idx].cuda()+identity2,mode='nearest').long()#.squeeze()

print(slic_train.shape,slic_val.shape)

identity2 = F.affine_grid(torch.eye(3,4).unsqueeze(0)+torch.randn(4,3,4)*.0,(4,1,96,80,128)).cuda()

#unroll loss for checkpointing
def slic_loss(output_j,slic,affine_j):
    with torch.no_grad():
        slic_warp = F.grid_sample(slic.float().unsqueeze(0).cuda(),affine_j,mode='nearest').long().squeeze(0)
    
    loss = 0
    for l in range(16):
        #correct!
        loss += 0.0625*nn.CrossEntropyLoss()(output_j.view(128,16,H//4,W//4,D//4)[:,l].unsqueeze(0),slic_warp[l:l+1,1::2,1::2,1::2])
    return loss

optimizer = torch.optim.Adam(list(backbone.parameters())+list(aspp.parameters())+list(head.parameters()),lr=0.004)
scaler = torch.cuda.amp.GradScaler()
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.1)
import time
t0 = time.time()
#training loop
run_loss = torch.zeros(5000)
run_val_loss = torch.empty(0,1)#
print('=== starting training ===')
t0 = time.time()
backbone.cuda(); backbone.train()
aspp.cuda(); aspp.train()
head.cuda(); head.train()
identity = F.affine_grid(torch.eye(3,4).unsqueeze(0).repeat(4,1,1),(4,1,192//4,160//4,256//4),align_corners=False)

#train for a couple thousand iterations (2000 seems to be enough)
for i in range(5000):
    

    #random mini-batch
    idx = torch.randperm(20)[:4]
    optimizer.zero_grad()
    #affine augmentation 
    with torch.no_grad():
        affine = F.affine_grid(torch.eye(3,4).unsqueeze(0)+torch.randn(4,3,4)*.07,(4,1,96,80,128)).cuda()
        identity2 = F.affine_grid(torch.eye(3,4).unsqueeze(0)+torch.randn(4,3,4)*.0,(4,1,96,80,128)).cuda()
        
        input = F.grid_sample(img_train[idx].cuda(),affine,padding_mode='border')#.flip(flips).flip(flips)
        input.requires_grad = True
        
    #forward path (including fp16 computation)
    with torch.cuda.amp.autocast():
        x1 = checkpoint(backbone[:3],input)
        x2 = checkpoint(backbone[3:],x1)
        y = checkpoint(aspp,x2)
        for j in range(4):
            #skip-connection
            y1 = torch.cat((x1[j:j+1],F.interpolate(y[j:j+1],scale_factor=2)),1)
            output_j = checkpoint(head,y1)
            loss += .25*checkpoint(slic_loss,output_j,slic_train[idx[j]],affine[j:j+1])


    #backward propagation
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    # Updates the scale for next iteration.
    scaler.update()
    #scheduler.step()
    #loss.backward()
    run_loss[i] = loss.item()
    #optimizer.step()
    if(i==1):
        print(i,time.time()-t0,'sec','loss',loss.item())
        gpu_usage()
    if(i%10==9):
        print(i,time.time()-t0,'sec','loss',loss.item())

        idx = torch.arange(4)
        with torch.no_grad():
            input = F.grid_sample(img_val[idx].cuda(),identity2,padding_mode='border')#.flip(flips).flip(flips)

            with torch.cuda.amp.autocast():
                x1 = checkpoint(backbone[:3],input)
                x2 = checkpoint(backbone[3:],x1)
                y = checkpoint(aspp,x2)
                val_loss = 0
                for j in range(4):
                    #print(j)
                    y1 = torch.cat((x1[j:j+1],F.interpolate(y[j:j+1],scale_factor=2)),1)
                    output_j = checkpoint(head,y1)
                    val_loss += .25*checkpoint(slic_loss,output_j,slic_val[idx[j]],identity2[:1])
        print(i,time.time()-t0,'sec','val_loss',val_loss)
        run_val_loss = torch.cat((run_val_loss,val_loss.data.cpu().view(-1,1)),0)

        gpu_usage()
        #validate_reg(img_val,seg_val1,pseudo_seg,backbone,aspp,head)
        #validate38(backbone,aspp,head,img_val,seg_val,fields_val,seg38)
        #backbone.cuda(); backbone.train()
        #aspp.cuda(); aspp.train()
        #head.cuda(); head.train()
    if(i%1000==499):
        print(i,'saving model')
        torch.save({'backbone':backbone.cpu().state_dict(),'aspp':aspp.cpu().state_dict(),'head':head.cpu().state_dict(),},\
          'mobile_aspp_slic16_iso_plus2.pth')
        backbone.cuda(); backbone.train()
        aspp.cuda(); aspp.train()
        head.cuda(); head.train()




#compute supervoxels for validation scans (used in adam reg)
output_val = torch.zeros(10,16,H//2,W//2,D//2).long()

for batch_ in range(2):
    if batch_==0:
        idx = torch.arange(0,5)
    else
        idx = torch.arange(5,10)
    t0 = time.time()
    val_case = torch.Tensor([1,4,7,10,23,26,29,32,35,38])
    print(val_case[idx])
    with torch.no_grad():
        identity2 = F.affine_grid(torch.eye(3,4).unsqueeze(0)+torch.randn(5,3,4)*.0,(5,1,96,80,128)).cuda()


        input = F.grid_sample(img_val[idx].cuda(),identity2,padding_mode='border')#.flip(flips).flip(flips)

        with torch.cuda.amp.autocast():
            x1 = checkpoint(backbone[:3],input)
            x2 = checkpoint(backbone[3:],x1)
            y = checkpoint(aspp,x2)
            val_loss = 0
            for j in range(5):
                #print(j)
                y1 = torch.cat((x1[j:j+1],F.interpolate(y[j:j+1],scale_factor=2)),1)
                y_ = checkpoint(head,y1).view(128,16,H//4,W//4,D//4)
                #correct:
                for k in range(16):
                    output_val[idx[j],k] = F.interpolate(y_[:,k:k+1],scale_factor=2,mode='trilinear').argmax(0)

    print(time.time()-t0,'sec','val_loss',val_loss)
    #output_val = F.interpolate(output_val.float(),scale_factor=2.0).long()

print(output_val.shape,output_val.max())
torch.save(output_val.byte(),'abdomen38_mobile_superisoplus_val2.pth')#.long()
 
