import torch
import pytorch3d.ops
import os
def sample_vertsparam(iteration_loc,vertsparam,bs,K=1024,random_start_point=False):
    global ix_of_sampled_vertsparam
    if iteration_loc == 0:
        
        sampled_vertsparam,ix_of_sampled_vertsparam = pytorch3d.ops.sample_farthest_points(    
            torch.cat([vertsparam.unsqueeze(0) for _ in range(bs)],dim=0), 
            lengths= None, K = K, 
            random_start_point = random_start_point
            )
        sampled_vertsparam = sampled_vertsparam.permute(0,2,1)
    else:
        cat_vertsparam = torch.cat([vertsparam.unsqueeze(0) for _ in range(bs)],dim=0)
        sampled_vertsparam = []
        for ix in range(bs):
            sampled_vertsparam.append(cat_vertsparam[ix][ix_of_sampled_vertsparam[ix]])
        sampled_vertsparam = torch.stack(sampled_vertsparam,dim=0)
        # sampled_vertsparam = cat_vertsparam[ix_of_sampled_vertsparam]
        sampled_vertsparam = sampled_vertsparam.permute(0,2,1)
    return sampled_vertsparam