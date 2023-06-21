import sys
import os
import pytorch3d.ops
import colorful
import torch
import importlib
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
sys.path.append('/root/evaluate-saliency-4/fong-invert/DeepInversion/dgcnn_pytorch')
from argparse import Namespace
from model import DGCNN_cls
def get_pointnet():
    pointnet_BASE_DIR = '/root/evaluate-saliency-4/fong-invert/point-radiance/Pointnet_Pointnet2_pytorch'
    sys.path.append(pointnet_BASE_DIR)
    sys.path.append(os.path.join(pointnet_BASE_DIR,'models'))
    log_dir = 'pointnet2_ssg_wo_normals'
    num_class = 40
    experiment_dir = 'log/classification/' + log_dir
    model_name = os.listdir(pointnet_BASE_DIR + '/'+ experiment_dir + '/logs')[0].split('.')[0]
    # assert False
    model = importlib.import_module(model_name)
    classifier = model.get_model(num_class,normal_channel=False)
    checkpoint = torch.load(pointnet_BASE_DIR + '/'+ str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    # print(pointcloud_ss.shape)
    # points  = pointcloud_ss.transpose(2,1)
    # print(all_label[sample_ix])
    # out = classifier(points)
    # print(out[0].argmax(dim=-1))
    return classifier
def get_dgcnn():
    # from dgcnn_model import DGCNN_cls
    dgcnn_args = Namespace()
    dgcnn_args.model_path = '/root/evaluate-saliency-4/fong-invert/point-radiance/dgcnn_pytorch2/pretrained/model.cls.1024.t7'
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    dgcnn_args.num_points = 1024
    dgcnn_args.k = 20 # number of nearest neighbors
    modelnet_model = DGCNN_cls(dgcnn_args).to(device)
    
    modelnet_model.eval()
    # import ipdb;ipdb.set_trace()
    return modelnet_model
class DGCNNInverter():
    def __init__(self):
        self.bs_modelnet = 1
        os.environ['DBG_FIXED_SAMPLED_POINTS'] = '1'
            
        print(colorful.yellow_on_red(
            'DBG_FIXED_SAMPLED_POINTS' + os.environ.get("DBG_FIXED_SAMPLED_POINTS",False)
            )
            )        
        pass
    def forward(self):
        sampled_vertsparam,ix_of_sampled_vertsparam = pytorch3d.ops.sample_farthest_points(    
            torch.cat([pr_model.vertsparam.unsqueeze(0) for _ in range(self.bs_modelnet)],dim=0), 
            lengths= None, K = 1024, 
            random_start_point = (True if not os.environ.get('DBG_FIXED_SAMPLED_POINTS',False) else False)
            )
        sampled_vertsparam = sampled_vertsparam.permute(0,2,1)        
        return sampled_vertsparam

import numpy as np
import glob
import h5py
def load_modelnet_data(
                        DATA_DIR = "/root/bigfiles/dataset",
                       partition = 'train'
                       ):
    
    all_data = []
    all_label = []
    
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data,all_label

# from modelnet_utils import load_modelnet_data
import pytorch3d.ops
import dutils

def viz_modelnet_label(label,
                       pr_args,
                        sample_ix = 1,
                       all_data = None,
                       all_label = None,
                       device = 'cuda'
                       ):
    from point_radiance_modules.model import CoreModel
    if all_data == None:
        all_data,all_label = load_modelnet_data()
    all_label = all_label.squeeze()
    ix_of_label = np.arange(len(all_label))[all_label == label]
    all_data = all_data[ix_of_label]
    all_label = all_label[ix_of_label]

    pointcloud = torch.tensor(all_data)[sample_ix:sample_ix + 1]
    pointcloud_ss,ixs_ss = pytorch3d.ops.sample_farthest_points(pointcloud,lengths = None,K= 1024, random_start_point= False)
    pointcloud_ss1 = pointcloud_ss[0]
    # pointcloud_ss1 = pointcloud_ss.permute(0,2,1).contiguous()
    pointcloud_ss1 = pointcloud_ss1.to(device)
    pr_model_modelnet = CoreModel(pr_args,STANDARD=False,init_mode='random').to(device)
    pr_model_modelnet.onlybase = True

    pr_model_modelnet.vertsparam = torch.nn.Parameter(pointcloud_ss1)
    pr_model_modelnet.sh_param = torch.nn.Parameter(torch.ones_like(pointcloud_ss1))
    pr_model_modelnet.to(device)
    from utils_3d import visualize_pr_model
    inputs_from_pr0 = visualize_pr_model(pr_model_modelnet,pr_args)
    dutils.img_save(inputs_from_pr0,f'viz_modelnet_{label}.png')
    
def get_modelnet(pretrained=True):
    from argparse import Namespace
    sys.path.append('/root/evaluate-saliency-4/fong-invert/DeepInversion/dgcnn_pytorch')
    from model import DGCNN_cls
    # from dgcnn_model import DGCNN_cls
    dgcnn_args = Namespace()
    dgcnn_args.model_path = '/root/evaluate-saliency-4/fong-invert/point-radiance/dgcnn_pytorch2/pretrained/model.cls.1024.t7'
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    dgcnn_args.num_points = 1024
    dgcnn_args.k = 20 # number of nearest neighbors
    net = DGCNN_cls(dgcnn_args)
    net = torch.nn.DataParallel(net)
    if pretrained:
        net.load_state_dict(torch.load(dgcnn_args.model_path))
    net = net.module
    net = net.eval()                
    return net
def get_order(gt_feats,feats):
    if gt_feats.ndim == 4:
        gt_feats0 = gt_feats.permute(0,2,3,1).flatten(start_dim=2,end_dim=3)
        feats0 = feats.permute(0,2,3,1).flatten(start_dim=2,end_dim=3)
        dist_matrix = ((feats0[:1][:,:,None,:] - gt_feats0[:,None,:,:])**2).sum(dim=-1)
    elif gt_feats.ndim == 3:
        # import ipdb;ipdb.set_trace()
        gt_feats0 = gt_feats.permute(0,2,1).flatten(start_dim=-1)
        feats0 = feats.permute(0,2,1).flatten(start_dim=-1)
        dist_matrix = ((feats0[:1][:,:,None,:] - gt_feats0[:,None,:,:])**2).sum(dim=-1)
    
    dist_matrix_ = tensor_to_numpy(dist_matrix)[0]
    """
    bipartite matching using scipy
    """
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(dist_matrix_)
    order = col_ind
    # order = dist_matrix.argmin(dim=-1)[0]
    # """
    # bipartite matching
    # """
    # order = torch.zeros_like(dist_matrix[0,0]).long()
    # remaining = np.arange(order.shape[0])
    # done = {}
    # for i in range(order.shape[0]):
    #     remaining = np.setdiff1d(remaining,list(done.keys()))
    #     o0 = dist_matrix[0,i,remaining].argmin(dim=-1)
    #     o = remaining[o0]
    #     order[i] = o
    #     done[o] = True
    return order

def get_interpoint_distance(sampled_vertsparam):
    interpoint_distance = torch.norm(sampled_vertsparam[:,:,None,:] - sampled_vertsparam[:,:,:,None],dim=1)
    # assert interpoint_distance.shape[-2:] == (locals().get('DBG_NPTS',1024) ,locals().get('DBG_NPTS',1024)) 
    interpoint_distance =interpoint_distance.mean().item()
    return interpoint_distance

def visualize_pointcloud(vertsparam,savename = '/tmp/dummy.png',c=None,AXIS_MAG = 1.2):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # color the points according to z axis
    if c is None:
        c= vertsparam[0,2,:]
    ax.scatter(vertsparam[0,0,:], vertsparam[0,1,:], vertsparam[0,2,:], c=c, marker='o',s = 5)
    # turn off grid and axis
    ax.grid(False)
    ax.axis('off')
    
    ax.set_zlim(-AXIS_MAG, AXIS_MAG)
    ax.set_xlim(-AXIS_MAG, AXIS_MAG)
    ax.set_ylim(-AXIS_MAG, AXIS_MAG)
    fig.subplots_adjust(0, 0, 1, 1)  # Set the margins to 0 on all sides

    # view distance
    
    plt.draw()
    
    plt.savefig(savename)

tensor_to_numpy = lambda tensor: tensor.detach().cpu().numpy()
from collections import defaultdict
class feature_tracker_():
    def __init__(self,enabled=True):
        self.gt = False
        self.losses = defaultdict(list)
        self.enabled = enabled
        pass
    def update(self,**kwargs):
        if not self.enabled:
            return
        for key,value in kwargs.items():
            if self.gt:
                self.__dict__['gt'+key] = tensor_to_numpy(value)
            else:
                if 'gt'+key in self.__dict__:
                    loss = ((value - self.__dict__['gt'+key])**2).mean()
                    self.losses[key].append(loss.item())
                    # import ipdb;ipdb.set_trace()
    pass
feature_tracker = feature_tracker_()

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx
