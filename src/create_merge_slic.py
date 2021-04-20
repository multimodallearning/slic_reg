import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.segmentation
import nibabel as nib
import cc3d

#applies skimage's SLIC 3d algorithm and merges the supervoxels to 128 elements
data_folder = 'AbdomenIsoCrop'
img = torch.from_numpy(nib.load(data_folder+'/Training/img/img0038.nii.gz').get_fdata()).float()/500
seg = torch.from_numpy(nib.load(data_folder+'/Training/label/label0038.nii.gz').get_fdata())
img = img.cuda()

from scipy.ndimage import distance_transform_edt as edt
seg38 = seg.cpu().clone()*1+0
sdt = edt(seg38==0)

#we first create 64 versions of supervoxels (and later use only 16 layers) because the merging sometimes fails to give the correct count
affine = torch.eye(3,4).unsqueeze(0)+torch.randn(64,3,4)*0.05
grid = F.affine_grid(affine,(64,1,96,80,128))#.cuda()
affine_inv = torch.inverse(torch.cat((affine,torch.Tensor([0,0,0,1]).view(1,1,4).repeat(64,1,1)),1))
grid_inv = F.affine_grid(affine_inv[:,:3,:],(64,1,96,80,128))#.cuda()
img_aff = torch.zeros(64,1,96,80,128)
sdt_aff = torch.zeros(64,1,96,80,128)

for i in range(64):
    img_aff[i] = F.grid_sample(img.unsqueeze(0).unsqueeze(0),grid[i:i+1].cuda(),padding_mode='border')
    sdt_aff[i] = F.grid_sample(torch.from_numpy(sdt).cuda().float().unsqueeze(0).unsqueeze(0),grid[i:i+1].cuda(),padding_mode='border')



torch.manual_seed(7)
#mask = torch.from_numpy(sdt<30).float()*

layers = torch.zeros(64,96,80,128)
import numpy as np
#takes a while!
for i in range(64):
    cluster = skimage.segmentation.slic(img_aff[i,0].cpu().numpy(),multichannel=False,mask=sdt_aff[i,0].cpu().numpy()<30,\
                                   compactness=1.5, enforce_connectivity=True, max_iter=5, n_segments=208, min_size_factor=0.85)
    #print('len unique',len(np.unique(cluster)))
    layers[i] = torch.from_numpy(cluster)#[:,35,:]==cluster[70,35,70]).float()
    
layers2 = torch.zeros_like(layers)

with torch.no_grad():
    for i in range(64):
        labels_out = torch.from_numpy(cc3d.connected_components(layers[i].int().cpu().numpy(), connectivity=26).astype('int64'))
        layers2[i] = F.grid_sample(labels_out.unsqueeze(0).unsqueeze(1).float().cuda(),grid_inv[i:i+1].cuda(),mode='nearest',padding_mode='border').squeeze().cpu()
        #print('unique',len(torch.unique(layers2[i])))
        #print((torch.bincount(layers2[i].long().view(-1))))

#this is fairly complicated, if you know an easier way let me know
def merge128(layers_int,grid_inv1):
    labels_out = cc3d.connected_components(layers_int.int().cpu().numpy(), connectivity=26)#.astype('int64')
    edges = cc3d.region_graph(labels_out, connectivity=26)
    edge_stack = torch.from_numpy(np.stack(edges))
    labels_out = torch.from_numpy(labels_out.astype('int64'))
    labels_out = F.grid_sample(labels_out.unsqueeze(0).unsqueeze(1).float().cuda(),grid_inv1,mode='nearest',padding_mode='border').squeeze().cpu().long()
        
    #get connected components and edge list
    #labels_out = cc3d.connected_components(layers_int.int().cpu().numpy(), connectivity=26)
    print('unique',len(np.unique(labels_out)))


    count = torch.bincount(labels_out.view(-1))
    merged_idx = torch.arange(1+(torch.max(labels_out)))
    #recursively merge smallest component with smallest connected one
    for i in range(190):
        min_count = torch.min(count[count>0])
        #print(min_count)
        #print(torch.arange(len(count))[count==min_count])
        idx_i = int(torch.arange(len(count))[count==min_count][0])
        #print(idx_i)
        candidates = torch.cat((edge_stack[edge_stack[:,0] == idx_i,1],edge_stack[edge_stack[:,1] == idx_i,0]),0)
        #print(candidates)
        count_ = count[candidates]
        min_count = torch.min(count_[count_>0])
        cand_idx = torch.arange(len(count_))[count_==min_count]
        cand_label = candidates[cand_idx]
        #print('self.count',count[idx_i],'cc.min.count',min_count,'out of',len(count_))
        #print('cand_idx',cand_idx,'label',cand_label)
        merged_idx[idx_i] = int(cand_label)
        count[int(cand_label)] += count[idx_i]
        count[idx_i] = 0
        #if((count>0).sum()==128+4):
        #    print('sum',count.sum(),'nonzero',(count>0).sum())
        if((count>0).sum()==128):
            break
    print('sum',count.sum(),'nonzero',(count>0).sum())

 
    #merge into consecutive integers
    labels_new = merged_idx[labels_out]
    unique_new = torch.unique(labels_new)
    unique_merge = torch.zeros(int(1+labels_new.max())).long()
    unique_merge[unique_new]=torch.arange(128)
    labels_new = unique_merge[labels_new]

    return labels_new
success = torch.zeros(64)
labels_new = torch.zeros(16,96,80,128).long()
count = 0
for i in range(64):
    try:
        labels_out = merge128(layers[i].cpu(),grid_inv[i:i+1].cuda())
        labels_new[count] = labels_out
        success[i] = 1
        count += 1
        print('success',len(torch.unique(labels_out.long().view(-1))),labels_out.shape)
        if(count==16):
            print('16 found')
            break
    except:
        success[i] = -1

print(success,(success*0.5+0.5).sum())
#print(torch.bincount(labels_new.view(-1)))        
print(labels_new.min(),labels_new.max())
torch.save(labels_new.byte().cpu(),'abdomenIsoCrop38_supervoxel16.pth')
!ls -l abdomenIsoCrop38_supervoxel16.pth

#don't forget to warp them with the baseline registration model
    #print(i)
