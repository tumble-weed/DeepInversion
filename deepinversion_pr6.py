# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
IGNORE_FOR_NOW = True
import pdb_attach
PDB_PORT = 50000
while True:
    try:
        pdb_attach.listen(PDB_PORT)  # Listen on port 50000.
        break
    except OSError:
        PDB_PORT += 1

import torch
import torch.nn as nn
import torch.optim as optim
import collections
# import torch.cuda.amp as amp
from apex import amp
import random
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import colorful
from utils.utils import lr_cosine_policy, lr_policy, beta_policy, mom_cosine_policy, clip, denormalize, create_folder
import my_utils
import os
import torch.nn.functional as F
import sys
import trace
from argparse import Namespace
from neural_point_prior import NPP,get_sphere
import modelnet_utils
import npp_hacks
import minor_experiments
import modelnet_utils
import pointcloud_models
import dutils
import augmentations_3d
tracer = trace.Trace(
    # ignoredirs=[sys.prefix, sys.exec_prefix],
    trace=0,
    count=1)

def hack(nms_dict,**kwargs):
    for k,v in kwargs.items():
        nms_dict[k] = v
        print(colorful.orange(f"setiing {k} {v}"))
def check_grad_mag(get_loss,*inputs):
    inputs2 =[]
    for i in inputs:
        i2 =i.detach().clone()
        i2 = i2.requires_grad_(True)
        inputs2.append(i2)
    
    get_loss(*inputs2).backward()
    return inputs2
# check_grad_mag(total_variation_3d_loss,pr_model.vertsparam[None], pr_model.sh_param[None]) 
mydir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(mydir)
PRDIR = os.path.join(parentdir, 'point-radiance')
sys.path.append(PRDIR)
from point_radiance_modules.model import CoreModel

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)              
from pytorch3d.structures import Pointclouds                  
torch.autograd.set_detect_anomaly(True)

tensor_to_numpy = lambda x: x.detach().cpu().numpy()
global opts
opts = my_utils.MyNamespace()
MOVEME = None
TODO = None
from collections import defaultdict
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        if os.environ.get('MAX_STD', False) == '1':
            r_feature = -torch.norm(var, 2)
        else:
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

class DeepInversionFeatureHook1d():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        ndim = input[0].ndim
        if ndim == 3:
            mean = input[0].mean([0, 2])
            var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False)
        elif ndim == 2:
            mean = input[0].mean([0])
            var = input[0].permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False)
        else:
            assert  False

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        if os.environ.get('MAX_STD', False) == '1':
            r_feature = -torch.norm(var, 2)
        else:
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()



# import torch
# import torch.nn.functional as F

def gaussian_blur(input_tensor, kernel_size, sigma=None):
    device =input_tensor.device
    if sigma is None:
        sigma = (kernel_size - 1) / 6  # Impute ideal sigma from kernel size

    channels = input_tensor.shape[1]  # Get number of channels from input tensor

    # Create Gaussian kernel
    kernel = torch.tensor([
        [(x - kernel_size // 2)**2 + (y - kernel_size // 2)**2 for x in range(kernel_size)]
        for y in range(kernel_size)
    ], dtype=torch.float,device=device)
    kernel = torch.exp(-kernel / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    # Apply Gaussian blur
    blurred = F.conv2d(input_tensor, kernel, stride=1, padding=kernel_size // 2, groups=channels)
    # import ipdb; ipdb.set_trace()
    return blurred


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    # print('check if var_l2 and var_l1 are dealing with the same ranges')
    # import ipdb; ipdb.set_trace()
    return loss_var_l1, loss_var_l2


class DeepInversionClass(object):
    def __init__(self, bs=84,
                 use_fp16=True, net_teacher=None, 
                 path="./gen_images_pr3/",
                 path_alias= None,
                 final_data_path="/gen_images_final/",
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display = None):
        '''
        :param bs: batch size per GPU for image generation
        :param use_fp16: use FP16 (or APEX AMP) for model inversion, uses less memory and is faster for GPUs with Tensor Cores
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param final_data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True, other options are not supported at this time
            "detach_student": if computing Adaptive DI, should we detach student?
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k
            2 - will run optimization on high resolution for 20k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L2 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        '''

        print("Deep inversion class generation")
        # for reproducibility
        if torch.cuda.is_available():
            torch.manual_seed(torch.cuda.current_device())
        else:
            torch.manual_seed(0)
        
        

        # self.net_teacher = net_teacher
        if 'MODELNET':
            import modelnet_utils
            
            if os.environ.get('MODELNET_MODEL','DGCNN') == 'POINTNET':
                modelnet_model = modelnet_utils.get_pointnet()
                # assert False
            elif os.environ.get('MODELNET_MODEL','DGCNN') == 'DGCNN':
                modelnet_model = modelnet_utils.get_modelnet(pretrained=True)
            # assert False
            # modelnet_model.to(device)        
        net_teacher = modelnet_model        
        self.net_teacher = modelnet_model        

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.startflip_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            assert False,'resolution should be sent in with params'
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.detach_student = False
            self.do_flip = True
            self.store_best_images = False

        self.setting_id = setting_id
        self.bs = bs  # batch size
        self.use_fp16 = use_fp16
        self.save_every = 100
        self.jitter = jitter
        self.criterion = criterion
        self.network_output_function = network_output_function
        do_clip = True

        if "r_feature" in coefficients:
            self.bn_reg_scale = coefficients["r_feature"]
            self.first_bn_multiplier = coefficients["first_bn_multiplier"]
            self.var_scale_l1 = coefficients["tv_l1"]
            self.var_scale_l2 = coefficients["tv_l2"]
            self.l2_scale = coefficients["l2"]
            self.lr = coefficients["lr"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
            self.adi_scale = coefficients["adi_scale"]
        else:
            print("Provide a dictionary with ")

        self.num_generations = 0
        self.final_data_path = final_data_path

        ## Create folders for images and logs
        prefix = path
        prefix_alias = path_alias
        self.prefix = prefix
        # self.prefix = self.prefix + '_dummy'
        
        npp_hacks.set_prefix(locals())
        
        self.prefix_alias = prefix_alias
        if torch.cuda.is_available():
            local_rank = torch.cuda.current_device()
        else:
            local_rank = 0 
        if local_rank==0:
            create_folder(prefix)
            create_folder(prefix + "/best_images/")
            create_folder(self.final_data_path)
            # save images to folders
            # for m in range(1000):
            #     create_folder(self.final_data_path + "/s{:03d}".format(m))

        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []
        # import ipdb;ipdb.set_trace()
        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))
            elif isinstance(module, nn.BatchNorm1d):
                # import ipdb;ipdb.set_trace()
                self.loss_r_feature_layers.append(DeepInversionFeatureHook1d(module))

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

    def get_images(self, net_student=None, targets=None):
        """
        class PlottableDict(dict):
            def __init__(self):
                self._dict = defaultdict(list)
                self.plottable = defaultdict(False)
                pass
            def __getitem__(self,key)
                return self._dict[key]
            def plottable(self,key):
                self.plottable[key] = True
            # def keys(self,):
            #     return self._dict.keys()
            # def items(self,):
            #     return self._dict.keys()
            # def keys(self,):
            #     return self._dict.keys()
        trends.plottable('')
        trends[''].append()

        """
        os.environ['MODELNET'] = '1'
        
        trends = defaultdict(list)
        dutils.trends = trends
        

        print("get_images call")
        if torch.cuda.is_available():
            device = "cuda:0"
            device_other = "cuda:1"
        else:
            device = "cpu"
        
        self.net_teacher.to(device)
        self.net_teacher.eval()
        
        
        use_fp16 = self.use_fp16
        # import ipdb; ipdb.set_trace()
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
        if torch.cuda.is_available():
            local_rank = torch.cuda.current_device()
        else:
            local_rank = 0 
        best_cost = 1e4
        criterion = self.criterion

        # setup target labels
        if 'MODELNET':
            # 'lamp' (confirm this is class 19)
            self.bs_modelnet = 1
            self.n_noise = 10
            # os.environ['USE_R_LOSS'] = '1'
            modelnet_targets = torch.tensor([17  for _ in range(self.bs_modelnet * self.n_noise) ],device=device)
            if os.environ.get('USE_R_LOSS',False) == '1':
                modelnet_targets = torch.tensor(range(self.bs_modelnet) ,device=device)
                modelnet_targets = modelnet_targets.repeat(self.n_noise)
            # imagenet 'table lamp'
            targets=  torch.tensor([846 for _ in range(self.bs)],device=device)

        data_type = torch.half if use_fp16 else torch.float

        #=======================================================================================
        do_clip = True
        from argparse import Namespace
        pr_args = Namespace()
        # pr_args.splatting_r = 0.015
        # pr_args.raster_n
        
        # pr_args.splatting_r = 0.015
        DEBUG_HIGH_SPLATTING_R = False
        DEBUG_LOW_SPLATTING_R = False
        pr_args.splatting_r = 0.02
        if DEBUG_LOW_SPLATTING_R:
            print(colorful.pink("low splatting r"))
            pr_args.splatting_r = 0.005        
        elif DEBUG_HIGH_SPLATTING_R:
            print(colorful.khaki("high splatting r"))
            pr_args.splatting_r = 0.1
        pr_args.raster_n = 15
        pr_args.refine_n = 2
        pr_args.data_r = 0.012
        pr_args.step = 'brdf'
        pr_args.savemodel = None

        
        pr_args.lr1 = float(os.environ.get('LR1',22e-3))
        pr_args.lr2 = float(os.environ.get('LR2',8e-4))
        pr_args.lrexp = 0.93
        pr_args.lr_s = 0.03
        # pr_args.img_s = img_original
        pr_args.img_s = 224
        pr_args.memitem = None


        pr_args.expname = 'pcdata'
        pr_args.basedir = '..'
        pr_args.datadir = 'nerf_synthetic'
        pr_args.dataname = 'hotdog'
        pr_args.grey = 1
                            

        # training options
        pr_args.netdepth = 8
        pr_args.netwidth = 256
        pr_args.netdepth_fine = 8
        pr_args.netwidth_fine = 256
        pr_args.N_rand = 32
        pr_args.lrate = 5e-4
        pr_args.lrate_decay = 500
        pr_args.chunk = 1024
        pr_args.netchunk = 1024
        pr_args.no_batching = True
        pr_args.no_reload = False
                            

        # rendering options
        pr_args.N_samples = 64
        pr_args.N_importance = 128
        pr_args.perturb = 1.
        pr_args.use_viewdirs = True
        pr_args.i_embed = 0
        pr_args.multires = 10 
        pr_args.multires_views = 4
        pr_args.raw_noise_std = 0.

        pr_args.render_only = False
        pr_args.render_test = False

        pr_args.render_factor = 0
                            

        # training options
        pr_args.precrop_iters = 500
                            
        pr_args.precrop_frac = .5

        # dataset options
        pr_args.dataset_type = 'blender'
        pr_args.testskip = 1
                            

        ## deepvoxels flags
        pr_args.shape = 'greek'

        ## blender flags
        pr_args.white_bkgd = False
        pr_args.half_res = True
                            

        ## llff flags
        pr_args.factor = 8
                            
        pr_args.no_ndc = True
        pr_args.lindisp = False
        pr_args.spherify = False

                            
        pr_args.llffhold = 8
                            

        # logging/saving options
        pr_args.i_print = 100
                            
        pr_args.i_img = 500
                            
        pr_args.i_weights = 500
                            
        pr_args.i_testset = 5000
                            
        pr_args.i_video = 10000
                                    
        #=======================================================================================
        """
        pr_model = CoreModel(pr_args,STANDARD=False,init_mode='random',npts=1024).to(device)
        pr_model.onlybase = True
        """
        pointcloud_model = pointcloud_models.get_point_model(device,locals(),n_mesh=self.bs_modelnet)
        my_utils.cipdb('DBG_PR')
        # END PR ----------------------------------------


        iteration = 0
        for lr_it, lower_res in enumerate([1]):

            iterations_per_layer = 2000                
            if self.setting_id == 2:
                iterations_per_layer = 20000
                
            if os.environ.get('EPOCHS',None):
                iterations_per_layer = int(os.environ['EPOCHS'])
                # import ipdb; ipdb.set_trace()                    

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
            
            # print('check what the setting_id is and if what happens if you remove color clipping?')


            """
            :param setting_id: predefined settings for optimization:
                0 - will run low resolution optimization for 1k and then full resolution for 1k;
                1 - will run optimization on high resolution for 2k
                2 - will run optimization on high resolution for 20k            
            """                 
            do_clip = True
            # import ipdb;ipdb.set_trace()
            import itertools
            if False:
                modelnet_optimizer = torch.optim.Adam([
                        {'params': itertools.chain(pointcloud_model.parameters(), [scale_logit]), 'lr': 1e-4},
                        ])
            else:
                modelnet_optimizer = torch.optim.Adam([
                        {'params': (pointcloud_model.parameters()), 
                        #  'lr': 1e-3*1e-2
                        # 'lr': 1e-3*1e0
                        # 'lr': 1e-3*1e-1
                        'lr': 1e-3
                         },
                        ])
            for iteration_loc in range(iterations_per_layer):
                iteration += 1


                # forward pass
                modelnet_optimizer.zero_grad()
                # R_cross classification loss
                # print('check if targets are just indicators')
                # import ipdb; ipdb.set_trace()





                # R_ADI

                if 'MODELNET':
                    """
                    sample points from point cloud
                    """
                    import pytorch3d.ops
                    
                    # os.environ['DBG_FIXED_SAMPLED_POINTS'] = '1'
                    if os.environ.get('DBG_FIXED_SAMPLED_POINTS','0') == '1':
                        print(colorful.yellow_on_red(
                        'DBG_FIXED_SAMPLED_POINTS' + os.environ.get("DBG_FIXED_SAMPLED_POINTS",'false')
                        )
                        )
                    
                    import sample_vertsparam
                    """
                    if pr_model.vertsparam.shape[0] == 1024:
                        sampled_vertsparam = pr_model.vertsparam[None,...]
                        sampled_vertsparam = sampled_vertsparam.permute(0,2,1)
                    else:
                        
                        sampled_vertsparam = sample_vertsparam.sample_vertsparam(
                            iteration_loc,pr_model.vertsparam,
                    self.bs_modelnet,K=1024,
                    random_start_point=(True if not os.environ.get('DBG_FIXED_SAMPLED_POINTS',False) else False))
                    """
                    n_ticks = 32
                    sphere = get_sphere(n_ticks,filled=False,mode='uniform',device=device)
                    sampled_sphere = sample_vertsparam.sample_vertsparam(
                    iteration_loc,sphere,
                    self.bs_modelnet,K=1024,
                    random_start_point=False)
                    sampled_sphere = sampled_sphere.permute(0,2,1)[0]
                    
                    if locals().get('USE_GENERATOR',False):
                        z = 4*torch.randn(self.bs_modelnet,pointcloud_model.inD).to(device)
                        pointcloud_6d = pointcloud_model(z)
                        pointcloud_6d = pointcloud_6d.permute(0,2,1)
                        vertsparam =pointcloud_6d[:,:3]
                        sampled_vertsparam = vertsparam                    
                        
                    elif os.environ['POINTCLOUD_TYPE'] == 'tree':
                        if iteration_loc == 0:
                            z_tree = 1*torch.randn(self.bs_modelnet,1,96).to(device)
                        vertsparam = pointcloud_model([z_tree]) 
                        # import ipdb;ipdb.set_trace()
                        vertsparam = vertsparam.permute(0,2,1)
                        sampled_vertsparam = vertsparam
                        print(colorful.yellow(f'{vertsparam.min().item()},{vertsparam.max().item()}'))
                    else:
                        if os.environ['POINTCLOUD_TYPE'] == 'mesh':
                            pointcloud_6d = pointcloud_model(None,n_noise=self.n_noise) 
                        elif os.environ['POINTCLOUD_TYPE'] == 'inn':
                            extras = {}
                            pointcloud_6d = pointcloud_model(sphere,extras=extras) 
                            
                        else:
                            pointcloud_6d = pointcloud_model(sampled_sphere)
                        vertsparam =pointcloud_6d[...,:3]
                        
                        if vertsparam.ndim == 2:
                            vertsparam = vertsparam[None,...]
                        vertsparam = vertsparam.permute(0,2,1)
                        if False:
                            vertsparam.retain_grad()
                        sampled_vertsparam = vertsparam
                        if False:
                            sampled_vertsparam = sampled_vertsparam.repeat(self.bs_modelnet,1,1)
                    DBG_NPTS = 1024
                    sampled_vertsparam = sampled_vertsparam[:,:,:DBG_NPTS]
                    if False:
                        vertsparam = vertsparam*torch.exp(0.1*scale_logit)

                    if False:
                        interpoint_distance = modelnet_utils.get_interpoint_distance(sampled_vertsparam)
                        trends['interpoint_distance'].append(interpoint_distance)
                    
                    

                    """
                    sampled_vertsparam = sample_vertsparam.sample_vertsparam(
                    iteration_loc,vertsparam,
                    self.bs_modelnet,K=1024,
                    random_start_point=(True if not os.environ.get('DBG_FIXED_SAMPLED_POINTS',False) else False))
                    """
                    # """

                    # """
                    #===============================

                    if False:
                        from losses_3d_inversion import get_knn_loss
                        knn_loss = get_knn_loss(sampled_vertsparam,20)
                    #===============================
                    if iteration_loc == 0:
                        def keep_feats(self,input,output):
                            self.feats = output
                            pass
                        if 'dgcnn' in str(self.net_teacher.__class__).lower():
                            # feat_layer = self.net_teacher.linear3
                            # feat_layer = self.net_teacher.graph_feat1
                            os.environ['LAYER'] = 'conv1'
                            # os.environ['LAYER'] = 'conv5'
                            # os.environ['LAYER'] = 'dp1'
                            # os.environ['LAYER'] = 'linear2'
                            
                            # feat_layer = list(self.net_teacher.conv1.modules())[0]
                        elif 'get_model' == (self.net_teacher.__class__.__name__):
                            #sa1,sa2,sa3,fc1,bn1,drop1,fc2,bn2,drop2,fc3
                            os.environ['LAYER'] = 'fc1'

                        feat_layer = self.net_teacher._modules[os.environ['LAYER']]
                        feat_layer.register_forward_hook(keep_feats)
                    
                    # import ipdb;ipdb.set_trace()
                    # self.net_teacher.eval()
                    assert not self.net_teacher.training
                    """
                    if 'running_sampled_vertsparam' not in self.__dict__:
                        self.running_sampled_vertsparam = sampled_vertsparam.detach()
                    self.running_sampled_vertsparam = 0.999*self.running_sampled_vertsparam.detach() + 0.001*sampled_vertsparam
                    """
                    os.environ['HACK_DGCNN_KNN'] = '0'
                    if iteration_loc == 0:
                        # knn_idx0 = modelnet_utils.knn(sampled_vertsparam,self.net_teacher.k)        
                        knn_idx0 = None                
                    if False and 'pointcloud_gt' in locals():
                        order0 = np.arange(pointcloud_gt.shape[0])
                        # shuffle order0
                        order0 = np.random.permutation(order0)
                        sampled_vertsparam = pointcloud_gt[None,order0,:].permute(0,2,1).repeat(self.bs_modelnet,1,1).contiguous()
                        print((sampled_vertsparam - pointcloud_gt[None,:,:].permute(0,2,1)).abs().sum())
                        # import ipdb;ipdb.set_trace()
                    # minor_experiments.overwrite_pointcloud_with_gt(locals())
                    # import ipdb;ipdb.set_trace()
                    
                    sampled_vertsparam1 = sampled_vertsparam
                    if os.environ['POINTCLOUD_TYPE'] not in  ['tree']:
                        """
                        sampled_vertsparam1 = sampled_vertsparam1 +0.1*torch.randn_like(sampled_vertsparam)
                        assert sampled_vertsparam1.shape[1:] == (3,DBG_NPTS)
                        """
                        sampled_vertsparam1 = augmentations_3d.add_noise(sampled_vertsparam1,noise_mag=0.1*0,adaptive=False,n_noise=(self.n_noise if sampled_vertsparam1.shape[0] < self.n_noise * self.bs_modelnet else None))
                    if 'USE_ROTATION' and False:
                        sampled_vertsparam1_ = sampled_vertsparam1
                        sampled_vertsparam1 = augmentations_3d.rotate_pointcloud(sampled_vertsparam1_,n_angles= sampled_vertsparam1_.shape[0])

                    modelnet_scores = self.net_teacher(
                        sampled_vertsparam1 ,
                        knn_idx0 = knn_idx0
                        )
                    feats = feat_layer.feats
                    modelnet_probs = torch.softmax(modelnet_scores,dim=-1)

                    score_loss = -modelnet_scores[torch.arange(modelnet_targets.shape[0],device=device),modelnet_targets].sum()
                    # import ipdb;ipdb.set_trace()
                    kl_loss = criterion(modelnet_scores,modelnet_targets).sum()
                    trends['score_loss'].append(score_loss.item())
                    trends['kl_loss'].append(kl_loss.item())
                    # loss_modelnet = score_loss
                    loss_modelnet = kl_loss
                    
                rescale = [self.first_bn_multiplier] + [ (1) for _ in range(len(self.loss_r_feature_layers)-1)]

                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])

                trends['loss_r_feature'].append(loss_r_feature.item())

                # trends['pr_acc'].append(pr_acc.item())
                #=========================================================
                
                if 'reconse loss':
                    if iteration_loc == 0:
                        pointcloud_gt = npp_hacks.get_modelnet_gt_aligned(sampled_vertsparam,modelnet_targets,sample_ix = 0,align=False)
                        pointcloud_gt = pointcloud_gt[:DBG_NPTS,:]
                        with torch.inference_mode():
                            F = self.net_teacher.use_cache
                            self.net_teacher.use_cache = False
                            os.environ['HACK_DGCNN_KNN'] = '0'
                            self.net_teacher.feature_tracker.gt = True
                            ref_scores = self.net_teacher(pointcloud_gt[None,:,:].permute(0,2,1))
                            self.net_teacher.feature_tracker.gt = False
                            self.net_teacher.use_cache = F
                            gt_feats = feat_layer.feats.detach().clone()
                            ref_probs = torch.softmax(ref_scores,dim=-1)
                            # import ipdb;ipdb.set_trace()
                            original_pred = ref_probs.argmax(dim=-1)
                            
                            original_target_prob = ref_probs[:,modelnet_targets[0]]
                    
                    if False:
                        loss_recons_hack = (pointcloud_gt[None,...].permute(0,2,1) - sampled_vertsparam).norm(dim=1).mean()

                    """
                    reorder gt_feats so that nearest gt_feats are matched with nearest sampled_vertsparam feats
                    """
                    # from modelnet_utils import get_order
                    loss_feats = torch.zeros(1,device=device).float()
                    # if locals().get('EXP_TYPE','FEATURE_MATCH') == 'FEATURE_MATCH':
                    if 'EXP_TYPE' in locals() and EXP_TYPE == 'FEATURE_MATCH':
                        if feats.ndim == 4:
                            # order = np.arange(gt_feats.shape[2])
                            order = modelnet_utils.get_order(gt_feats,feats[:1])
                            loss_feats = ((feats - gt_feats[:,:,order,:])**2).mean()
                        elif feats.ndim == 3:
                            order = modelnet_utils.get_order(gt_feats,feats[:1])
                            loss_feats = ((feats - gt_feats[:,:,order])**2).mean()
                        elif feats.ndim == 2:
                            loss_feats = ((feats - gt_feats)**2).mean()
                    # import ipdb;ipdb.set_trace()
                    trends['loss_feats'].append(loss_feats.item())
                    print(modelnet_probs[torch.arange(modelnet_targets.shape[0],device=device),modelnet_targets].mean(),modelnet_probs.argmax(dim=-1),
                          modelnet_probs[:,modelnet_targets].max(),
                          modelnet_probs.max(),
                          original_pred,
                          original_target_prob,
                          )                    
                #=========================================================

                loss_l2_vertsparam = vertsparam.norm(dim=1).mean() 
                vertsparam_flatbc = vertsparam.permute(0,2,1).reshape(-1,3)
                loss_l2_vertsparam_1 = vertsparam_flatbc[vertsparam_flatbc.norm(dim=1) > 1].norm(dim=1).mean()
                # import ipdb;ipdb.set_trace()
                trends['loss_l2_vertsparam_1'].append(loss_l2_vertsparam_1.item())
                
                EXP_TYPE = 'FEATURE_MATCH'
                # EXP_TYPE = 'CLASS_MAXIMIZE'
                EXP_TYPE = os.environ.get('EXP_TYPE',EXP_TYPE)
                if iteration_loc == 0:
                    npp_hacks.set_optimizer_lr(modelnet_optimizer,locals(),EXP_TYPE)                
                if EXP_TYPE == 'CLASS_MAXIMIZE':
                    loss = loss_modelnet
                elif EXP_TYPE == 'FEATURE_MATCH':
                    # loss =  0.0001*loss_feats + 1*loss_modelnet
                    loss =  1*loss_feats + 0*loss_modelnet
                # loss = loss_modelnet = loss_recons_hack
                if os.environ['POINTCLOUD_TYPE'] == 'inn':
                    loss = 0.01*loss + 100*((extras['logdet'])**2).sum() #- 1e-6*extras['logdet'].sum() +loss_l2_vertsparam*0.00001
                if os.environ['POINTCLOUD_TYPE'] == 'pr':
                    loss = loss + 0.1*loss_l2_vertsparam+ 100*0.0001*loss_r_feature
                if os.environ['POINTCLOUD_TYPE'] == 'sh':
                    coeff_decay = 0.1*((pointcloud_model.coeffs[1:])**2).sum()
                    loss = loss + 0.1*loss_l2_vertsparam
                    # loss = loss + 0.1*coeff_decay
                    # assert False
                if os.environ['POINTCLOUD_TYPE'] == 'tree':
                    weight_term = 0
                    for mod in pointcloud_model.gcn:
                        for mod2 in mod.modules():
                            for w in mod2.parameters():
                                if w.ndim > 0:  
                                    weight_term = weight_term + (w*w).sum()
                    loss = loss  + weight_term
                if os.environ['POINTCLOUD_TYPE'] == 'mesh':
                    mesh_losses = pointcloud_model.get_prior_losses()
                    if EXP_TYPE == 'CLASS_MAXIMIZE':
                        if pointcloud_model.levels == 3:
                            assert False
                            loss = 1e-1*(1/100)*100*(loss + 0*(1/10)*100*mesh_losses['edge'] + 0*mesh_losses['laplacian'] + 0*1*mesh_losses['normal']) 
                            # assert False
                            if os.environ.get('USE_R_LOSS',False) == '1':
                                loss = loss + 1000*loss_r_feature
                        else:
                            """
                            from argparse import Namespace
                            factors = Namespace()
                            factors.edge_factor = 0*0.1
                            factors.r_factor = 0.0001
                            factors.loss_factor = 1
                            """
                            import loss_factors
                            # factors = loss_factors.score_edge_and_normal
                            # factors = loss_factors.score_and_edge
                            # factors = loss_factors.just_score
                            # factors = loss_factors.score_and_r
                            # factors = loss_factors.score_r_and_edge
                            factors = loss_factors.score_r_edge_and_normal
                            # locals().update(loss_factors.score_and_r)
                            loss = (factors.loss_factor*loss/self.n_noise 
                                    
                                    + factors.edge_factor*mesh_losses['edge'] 
                                    + 0*mesh_losses['area']
                                    + 0*1*mesh_losses['laplacian'] 
                                    + factors.normal_factor*mesh_losses['normal'] 
                                    + factors.r_factor*loss_r_feature
                                    + loss_l2_vertsparam_1
                                    )
                            # assert False

                        # assert False
                    elif EXP_TYPE == 'FEATURE_MATCH':
                        assert False
                        loss = loss + 0*1*mesh_losses['edge'] + 0*1*mesh_losses['laplacian'] + 0.1*mesh_losses['normal'] 
                    # assert False
                    trends['edge'].append(mesh_losses['edge'].item())
                    trends['laplacian'].append(mesh_losses['laplacian'].item())
                    trends['normal'].append(mesh_losses['normal'].item())
                    trends['area'].append(mesh_losses['area'].item())
                trends['loss'].append(loss.item())
                trends['loss_modelnet'].append(loss_modelnet.item())
                trends['loss_l2_vertsparam'].append(loss_l2_vertsparam.item())
                
                #=========================================================

                if local_rank==0:
                    if iteration % save_every==0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        # print("main criterion", criterion(outputs, targets).mean().item())

                        # if self.hook_for_display is not None:
                        #     self.hook_for_display(inputs, targets)

                # do image update
                loss.backward()
                # import ipdb;ipdb.set_trace()
                modelnet_optimizer.step()   
                
                if do_clip:
                    if os.environ['POINTCLOUD_TYPE'] == 'pr':
                        vertparam = pointcloud_model.vertparam.detach().clone()
                        # vertsparam = pr_model.vertsparam.detach().clone()
                        assert vertparam.shape[-1] == 3
                        lengths = vertparam.norm(dim=-1)[...,None]
                        lengths[lengths<1] = 1
                        pointcloud_model.vertparam.data.copy_(
                            # vertsparam.data.clamp(-1,1)
                            vertparam/lengths
                        )
                        assert pointcloud_model.vertparam.norm(dim=-1).max() <= 1.0001
                        # assert (pr_model.sh_param <= 1.).all()
                    else:   
                        print(colorful.chartreuse("skipping clipping vertsparams"))

                # clip color outlayers
                if not IGNORE_FOR_NOW:
                    if best_cost > loss.item() or iteration == 1:
                        best_inputs = inputs.data.clone()
                        best_cost = loss.item()
                print(colorful.orange(f"{self.prefix}"))
                print(colorful.orange(f"{PDB_PORT}"))
                print(colorful.red('make losses with surface_z'))
                # import ipdb; ipdb.set_trace()                
                
                if (save_every > 0)  and any([iteration % save_every==0  , 
                                            #   iteration in [10,20,30,40,50,60,70,80,90]
                                              ]) or os.environ.get('SAVE_ALL',False) == '1' :
                    #or USE_TRANSPARENCY:
                    if os.environ.get('SAVE_ALL',False) == '1':
                        import ipdb; ipdb.set_trace()
                    if local_rank==0:
                        print(self.prefix)
                        # assert False
                        # import ipdb; ipdb.set_trace()
                        for lname,loss_to_plot in self.net_teacher.feature_tracker.losses.items():
                            to_plot = loss_to_plot
                            # if lname == 'loss_modelnet':
                            #     to_plot = np.log(np.array(trends[lname])+1)
                            my_utils.save_plot(to_plot,lname+'_loss',f'{os.path.join(self.prefix,"best_images",lname+"_loss")}.png')                            
                        for lname in [
                            'loss_r_feature',
                            # 'pr_acc',
                            'loss_l2_vertsparam',#'loss_l2_surface_z',
                            #'loss_var_l2_surface_z',
                            'loss_modelnet',
                            'loss_feats',
                            'loss',
                            'interpoint_distance',
                            
                            'edge',
                            'laplacian',
                            'normal',
                            'area',
                            'grad_through_prob',
                            'grad_through_x',
                            'score_loss',
                            'kl_loss',
                            ]:
                            to_plot = trends[lname]
                            # if lname == 'loss_modelnet':
                            #     to_plot = np.log(np.array(trends[lname])+1)
                            my_utils.save_plot(to_plot,lname,f'{os.path.join(self.prefix,"best_images",lname)}.png')
                            # my_utils.save_plot(trends[lname],lname,f'{lname}.png')



                        
                        """
                        pr_model = CoreModel(pr_args,STANDARD=False,init_mode='random').to(device)
                        pr_model.onlybase = True
                        pr_model.vertsparam = None
                        pr_model.vertsparam = torch.nn.Parameter(sampled_vertsparam[0].permute(1,0))
                        pr_model.sh_param = torch.nn.Parameter(torch.ones_like(pr_model.vertsparam))

                        from utils_3d import visualize_pr_model
                        inputs_from_pr0 = visualize_pr_model(pr_model,pr_args)
                        # import ipdb;ipdb.set_trace()
                        
                        vutils.save_image(inputs_from_pr0,
                                          outputpath_pr,
                                          normalize=False, scale_each=False, nrow=int(10))           
                        """
                        # to_viz = sampled_vertsparam - sampled_vertsparam.mean(dim=1,keepdim=True)
                        # to_viz = to_viz/to_viz.abs().max()
                        # to_viz = sampled_vertsparam[17:18]
                        if os.environ['POINTCLOUD_TYPE'] == 'mesh':
                            pointcloud_6d = pointcloud_model(None,n_noise=self.n_noise,n_points_to_sample=1024) 
                            vertsparam =pointcloud_6d[...,:3]
                            sampled_vertsparam = vertsparam.permute(0,2,1)
                        if os.environ['POINTCLOUD_TYPE'] == 'inn':
                            extras = {}
                            pointcloud_6d = pointcloud_model(sphere,extras=extras) 
                            vertsparam =pointcloud_6d[...,:3]
                            if vertsparam.ndim == 2:
                                vertsparam = vertsparam[None,...]
                            sampled_vertsparam = vertsparam.permute(0,2,1)                            
                        to_viz = sampled_vertsparam
                        
                        #=============================================
                        outputpath_pr = '{}/best_images/output_pr_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                            iteration // save_every,
                                                                                            local_rank)
                        modelnet_utils.visualize_pointcloud(tensor_to_numpy(to_viz),savename=outputpath_pr)      
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","output_pr_latest.png")}')
                        os.system(f'ln -s {os.path.abspath(outputpath_pr)} {os.path.join(self.prefix,"best_images","output_pr_latest.png")}') 
                        #=============================================
                                                
                        if os.environ['POINTCLOUD_TYPE'] == 'mesh':
                            modelnet_utils.visualize_pointcloud(tensor_to_numpy(pointcloud_model.deform()[0][None,...].permute(0,2,1)),savename=f'{self.prefix}/best_images/output_pr_mesh.png')
                            
                            import pickle
                            with open(f'{self.prefix}/best_images/output_pr_mesh.pkl','wb') as f:
                                pickle.dump(
                                    [tensor_to_numpy(pc) for pc in pointcloud_model.deform()]
                                    ,f)

                        if False:

                            outputpath_pr_grad = '{}/best_images/output_pr_{:05d}_gpu_{}_grad.png'.format(self.prefix,
                                                        iteration // save_every,local_rank)
                            g = vertsparam.grad
                            if g.ndim == 2:
                                g = g[None,...]                        
                            gcolor = tensor_to_numpy(g.abs().sum(dim=1))
                            gcolor = gcolor.mean(axis=0,keepdims=True)
                            gcolor = gcolor/gcolor.max(axis=-1)[:,None]
                            non_zero_grad = gcolor > 0

                            modelnet_utils.visualize_pointcloud(tensor_to_numpy(sampled_vertsparam[:1,:,np.reshape(non_zero_grad,-1)]),savename= outputpath_pr_grad,c=gcolor[non_zero_grad])
                            # import ipdb;ipdb.set_trace()
                            os.system(f'unlink {os.path.join(self.prefix,"best_images","output_pr_latest_grad.png")}')                        
                            os.system(f'ln -s {os.path.abspath(outputpath_pr_grad)} {os.path.join(self.prefix,"best_images","output_pr_latest_grad.png")}') 
                        
                        #=======================================================
                        if iteration_loc < 2 * save_every:
                            from modelnet_utils import viz_modelnet_label
                            viz_modelnet_label(modelnet_targets[0].item(),
                                                pr_args,
                                                    sample_ix = 0,
                                                all_data = None,
                                                all_label = None,
                                                #    device = 'cpu'
                                                )
                        
                        #=======================================================
 

                        if iteration == save_every:
                            
                            if os.system(f'ln -s {self.prefix} {self.prefix_alias}') != 0:
                                os.system(f'rm {self.prefix_alias}')
                                
                                os.system(f'ln -s {os.path.abspath(self.prefix)} {self.prefix_alias}')


        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer we deallocate memory

        modelnet_optimizer.state = collections.defaultdict(dict)

    def save_images(self, images, targets):
        # method to store generated images locally
        if torch.cuda.is_available():
            local_rank = torch.cuda.current_device()
        else:
            local_rank = 0
            
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if 0:
                #save into separate folders
                place_to_store = '{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                          self.num_generations, id,
                                                                                          local_rank)
            else:
                place_to_store = '{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                          self.num_generations, id,
                                                                                          local_rank)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self, net_student=None, targets=None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # for ADI detach student and add put to eval mode
        net_teacher = self.net_teacher

        use_fp16 = self.use_fp16

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()

        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).to(device)
            if use_fp16:
                targets = targets.half()

        self.get_images(net_student=net_student, targets=targets)

        net_teacher.eval()

        self.num_generations += 1
