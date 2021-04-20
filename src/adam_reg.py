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
##### kpts / graph #####

def kpts_pt(kpts_world, shape, align_corners=None):
    dtype = kpts_world.dtype
    device = kpts_world.device
    D, H, W = shape
   
    kpts_pt_ = (kpts_world.flip(-1) / (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    
    return kpts_pt_

def kpts_world(kpts_pt, shape, align_corners=None):
    dtype = kpts_pt.dtype
    device = kpts_pt.device
    D, H, W = shape
    
    if not align_corners:
        kpts_pt /= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    kpts_world_ = (((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)).flip(-1) 
    
    return kpts_world_

def flow_pt(flow_world, shape, align_corners=None):
    dtype = flow_world.dtype
    device = flow_world.device
    D, H, W = shape
    
    flow_pt_ = (flow_world.flip(-1) / (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)) * 2
    if not align_corners:
        flow_pt_ *= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    
    return flow_pt_

def flow_world(flow_pt, shape, align_corners=None):
    dtype = flow_pt.dtype
    device = flow_pt.device
    D, H, W = shape
    
    if not align_corners:
        flow_pt /= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    flow_world_ = ((flow_pt / 2) * (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)).flip(-1)
    
    return flow_world_

def random_kpts(mask, d, num_points=None):
    _, _, D, H, W = mask.shape
    device = mask.device
    
    kpts = torch.nonzero(mask[:, :, ::d, ::d, ::d]).unsqueeze(0).float()[:, :, 2:]
    
    if not num_points is None:
        kpts = kpts[:, torch.randperm(kpts.shape[1])[:num_points], :]
    
    return kpts_pt(kpts, (D//d, H//d, W//d))

def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device
    
    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    
    return ind, dist*A, A

def lbp_graph(kpts_fixed, k):
    device = kpts_fixed.device
    
    A = knn_graph(kpts_fixed, k, include_self=False)[2][0]
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A).long()
    edges_idx[A.bool()] = torch.arange(edges.shape[0]).to(device)
    edges_reverse_idx = edges_idx.t()[A.bool()]
    
    return edges, edges_reverse_idx

class TPS:       
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device
        
        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device)*lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n+4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n+4, n+4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.solve(v, A)[0]
        return theta
        
    @staticmethod
    def d(a, b):
        ra = (a**2).sum(dim=1).view(-1, 1)
        rb = (b**2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r**2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + a[3]*x[:, 2] + b.t()).t()
    
def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_factor=100):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D//step, H//step, W//step
    
    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1)).view(-1,3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)
    
    y2 = torch.zeros((1, D1*H1*W1, 3), device=device)
    split = np.array_split(np.arange(D1*H1*W1), unroll_factor)
    for i in range(unroll_factor):
        y2[0, split[i], :] = tps.z(x2[split[i]], x1[0], theta)
    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear').permute(0, 2, 3, 4, 1)
    
    return y2
#   

##### distance #####

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist


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

print('case',ix)
  #convert argmax prediction to one_hot (with "compressed channels")

  with torch.no_grad():
      for i in range(16):
          for j in range(8):
              #print(to32[i,j*32:(j+1)*32].shape,torch.randperm(32).shape)
              to32[i,j*16:(j+1)*16] = torch.randperm(16)
          slic32[i] = to32[i][slic[i]]
          slic_warp32[i] = to32[i][slic_pred[ix][i]]







