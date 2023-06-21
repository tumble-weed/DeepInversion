import torch
import numpy as np
def rotate_pointcloud(points_bcp,n_angles=None,flip=True):
    device = points_bcp.device
    n_angles = points_bcp.shape[0]
    centers = points_bcp.mean(dim=-1,keepdim=True)
    centered = points_bcp - centers
    phi,delta = torch.rand(n_angles)*2*np.pi,torch.rand(n_angles,device=device)*np.pi
    rotated_points_bcp = []
    for i,(phi_i,delta_i) in enumerate(zip(phi,delta)):
        rotated_centered_i = centered[i:i+1].permute(0,2,1).clone()
        # if flip:
        #     # takre mirror image of the point cloud
        #     flipi =  torch.rand(1) > 0.5
        #     if flipi:
        #         rotated_centered_i = rotated_centered_i * torch.tensor([-1,1,1],device=device)[None,None,:]

        R = torch.tensor([[torch.cos(phi_i),-torch.sin(phi_i),0],
                          [torch.sin(phi_i),torch.cos(phi_i),0],
                          [0,0,1]],device=device)
        # print(rotated_points_bcp_i.shape)
        rotated_centered_i = torch.matmul(rotated_centered_i,R)
        R = torch.tensor([[torch.cos(delta_i),0,torch.sin(delta_i)],
                          [0,1,0],
                          [-torch.sin(delta_i),0,torch.cos(delta_i)]],device=device)
        rotated_centered_i = torch.matmul(rotated_centered_i,R)
        # print(rotated_points_bcp_i.shape)
        rotated_points_bcp.append(rotated_centered_i + centers[i:i+1].permute(0,2,1))
    rotated_points_bcp = torch.cat(rotated_points_bcp,dim=0)
    rotated_points_bcp = rotated_points_bcp.permute(0,2,1)
    # import ipdb;ipdb.set_trace()
    return rotated_points_bcp

        
def add_noise(sampled_vertsparam,noise_mag=0.1,adaptive=False,n_noise=None):
    device = sampled_vertsparam.device
    noise_shape = list(sampled_vertsparam.shape)
    if n_noise is not None:
        noise_shape[0] = noise_shape[0] * n_noise
    noise_shape = tuple(noise_shape)
    if adaptive:
        # sampled_vertsparam1 = sampled_vertsparam1 +0.1*sampled_vertsparam.permute(0,2,1).reshape(-1,3).std(dim=0)[None,:,None]*torch.randn_like(sampled_vertsparam)        
        noise = noise_mag*sampled_vertsparam.permute(0,2,1).reshape(-1,3).std(dim=0)[None,:,None]*torch.randn(noise_shape,device=device)
    else:
        noise = noise_mag*torch.randn(noise_shape,device=device)
    sampled_vertsparam1 = sampled_vertsparam + noise
    # assert sampled_vertsparam1.shape[1:] == (3,DBG_NPTS)
    assert sampled_vertsparam1.shape[1] == 3
    return sampled_vertsparam1
    
    