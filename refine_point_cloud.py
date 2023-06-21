#============================================================================  
import torch
import pytorch3d.ops
from pytorch3d.ops import knn_points
from point_radiance_modules.utils import remove_outlier              
def repeat_pts(vertsparam,sh_param,target_n_vertices=None):
    # import ipdb; ipdb.set_trace()
    to_repeat_vertsparam = vertsparam
    to_repeat_sh_param = sh_param
    
    if target_n_vertices is not None:
        K = target_n_vertices - vertsparam.shape[0]
        # import ipdb; ipdb.set_trace()
        pointcloud_ss,ixs_ss = pytorch3d.ops.sample_farthest_points(vertsparam.unsqueeze(0),lengths = None,K= K, random_start_point= False)
        ixs_ss = ixs_ss.squeeze()
        to_repeat_vertsparam = vertsparam[ixs_ss]
        to_repeat_sh_param = sh_param[ixs_ss]
        vertsparam.data = torch.cat([vertsparam,to_repeat_vertsparam],dim=0).data
        sh_param.data = torch.cat([sh_param,to_repeat_sh_param],dim=0).data
    else:
        vertsparam.data = to_repeat_vertsparam.data.repeat(2,1)
        sh_param.data = to_repeat_sh_param.data.repeat(2, 1)
    if vertsparam.grad is not None:
        vertsparam.grad = vertsparam.grad.repeat(2,1)
    if sh_param.grad is not None:
        sh_param.grad = sh_param.grad.repeat(2, 1)

def remove_out(vertsparam,sh_param):
    # import ipdb; ipdb.set_trace()
    pts_all = vertsparam.data
    pts_in = remove_outlier(pts_all.cpu().data.numpy())
    pts_in = torch.tensor(pts_in).cuda().float()
    idx = knn_points(pts_in[None,...], pts_all[None,...], None, None, 1).idx[0,:,0]
    vertsparam.data = vertsparam.data[idx].detach()
    sh_param.data = sh_param.data[idx].detach()
    if vertsparam.grad is not None:
        vertsparam.grad = vertsparam.grad[idx].detach()
    if sh_param.grad is not None:
        sh_param.grad = sh_param.grad[idx].detach()   