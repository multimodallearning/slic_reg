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
