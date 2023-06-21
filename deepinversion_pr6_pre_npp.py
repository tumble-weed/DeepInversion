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
def set_optimizer(
                # pr_model, 
                sh_params,
                other_params,
                  lr1=3e-3, lr2=8e-4,lrexp=0.93,lr_s=0.03):
    # sh_list = [name for name, params in pr_model.named_parameters() if 'sh' in name]
    # sh_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in sh_list,
    #                         pr_model.named_parameters()))))
    # other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in sh_list,
    #                         pr_model.named_parameters()))))
    optimizer = torch.optim.Adam([
        {'params': sh_params, 'lr': lr1},
        {'params': other_params, 'lr': lr2}])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lrexp, -1)
    # self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
    return optimizer,lr_scheduler
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
            modelnet_model = DGCNN_cls(dgcnn_args)
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
        print(colorful.yellow_on_red('setting DGNN to True'))        
        trends = defaultdict(list)

        ENABLE_PR = float(os.environ.get('ENABLE_PR',1))

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
            modelnet_targets = torch.tensor([19 ],device=device)
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
        pr_model = CoreModel(pr_args,STANDARD=False,init_mode='random',npts=1024).to(device)
        pr_model.onlybase = True
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
            modelnet_optimizer = torch.optim.Adam([
                    {'params': pr_model.vertsparam, 'lr': 1e-1},
                    ])
       
            # PR ---------------------------------------------
            pr_model.train()
            # END PR---------------------------------------------

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
                    self.bs_modelnet = 10
                    # os.environ['DBG_FIXED_SAMPLED_POINTS'] = '1'
                    
                    print(colorful.yellow_on_red(
                        'DBG_FIXED_SAMPLED_POINTS' + os.environ.get("DBG_FIXED_SAMPLED_POINTS",'false')
                        )
                        )
                    import sample_vertsparam
                    if pr_model.vertsparam.shape[0] == 1024:
                        sampled_vertsparam = pr_model.vertsparam[None,...]
                        sampled_vertsparam = sampled_vertsparam.permute(0,2,1)
                    else:
                        
                        sampled_vertsparam = sample_vertsparam.sample_vertsparam(
                            iteration_loc,pr_model.vertsparam,
                    self.bs_modelnet,K=1024,
                    random_start_point=(True if not os.environ.get('DBG_FIXED_SAMPLED_POINTS',False) else False))

                    
                    #===============================
                    def knn(x, k):
                        inner = -2*torch.matmul(x.transpose(2, 1), x)
                        xx = torch.sum(x**2, dim=1, keepdim=True)
                        pairwise_distance = -xx - inner - xx.transpose(2, 1)
                    
                        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
                        return idx
                    
                    from losses_3d_inversion import get_knn_loss
                    knn_loss = get_knn_loss(sampled_vertsparam,20)
                    #===============================
                    
                    # import ipdb;ipdb.set_trace()
                    modelnet_scores = self.net_teacher(sampled_vertsparam + 0e0*torch.randn(sampled_vertsparam.shape,device=device))
                    modelnet_probs = torch.softmax(modelnet_scores,dim=-1)
                    print(modelnet_probs[:,modelnet_targets].mean(),modelnet_probs.argmax(dim=-1))
                    if False:
                        loss_modelnet = -modelnet_scores[:,modelnet_targets].sum()
                    else:
                        # loss_modelnet = criterion(modelnet_scores,modelnet_targets.repeat(self.bs_modelnet)).sum()
                        loss_modelnet = criterion(modelnet_scores,modelnet_targets).sum()
                    # import ipdb;ipd.set_trace()
                    pass
                rescale = [self.first_bn_multiplier] + [ (1) for _ in range(len(self.loss_r_feature_layers)-1)]

                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])

                """
                if len(trends['pr_acc']) and  max(trends['pr_acc']) == 1 and pr_acc < 0.1:
                    for lname in ['pr_acc','main_loss_pr','main_loss_other','mesh_volume','n_pts','dist_min','loss_var_l2','loss_l2_vertsparam','loss_l2_pr','loss_var_l2_masks','loss_var_l2_zbuf','loss_var_l2_first','tv_3d','loss_l2_masks','loss_l2_zbuf','loss_modelnet']:
                        my_utils.save_plot(trends[lname],lname,f'{os.path.join(self.prefix,"best_images",lname)}.png')                    
                """
                # trends['pr_acc'].append(pr_acc.item())
                """
                next? actual classification oss?
                """     
                os.environ['INVERT_MODELNET'] = '1'
                print(colorful.yellow_on_red('hard coding INVERT_MODELNET'))
                #=========================================================
                # import ipdb;ipdb.set_trace()
                loss_l2_vertsparam = pr_model.vertsparam.norm(dim=1).mean() 
                loss = loss_modelnet+ knn_loss + 0.01*loss_l2_vertsparam+ 0*0.0001*loss_r_feature
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

                    if False:
                        vertsparam = pr_model.vertsparam.detach().clone()
                        lengths = vertsparam.norm(dim=-1)[:,None]
                        lengths[lengths<1] = 1
                        pr_model.vertsparam.data.copy_(
                            # vertsparam.data.clamp(-1,1)
                            vertsparam/lengths
                        )
                        assert pr_model.vertsparam.norm(dim=-1).max() <= 1.0001
                        assert (pr_model.sh_param <= 1.).all()
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

                        # import ipdb; ipdb.set_trace()
                        
                        for lname in [
                            # 'pr_acc',
                            'loss_l2_vertsparam',#'loss_l2_surface_z',
                            #'loss_var_l2_surface_z',
                            'loss_modelnet'
                            ]:
                            my_utils.save_plot(trends[lname],lname,f'{os.path.join(self.prefix,"best_images",lname)}.png')
                            # my_utils.save_plot(trends[lname],lname,f'{lname}.png')

                        from utils_3d import visualize_pr_model
                        inputs_from_pr0 = visualize_pr_model(pr_model,pr_args)
                        # import ipdb;ipdb.set_trace()
                        outputpath_pr = '{}/best_images/output_pr_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank)              
                        vutils.save_image(inputs_from_pr0,
                                          outputpath_pr,
                                          normalize=False, scale_each=False, nrow=int(10))           
                        """
                        maskpath = '{}/best_images/mask_pr_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank)
                        vutils.save_image(masks_from_pr,
                                          maskpath,
                                          normalize=False, scale_each=False, nrow=int(10))           

                        """
                        


                            
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","output_latest.png")}')
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","output_pr_latest.png")}')                        
                        """
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","mask_pr_latest.png")}')
                        """

                        # import ipdb; ipdb.set_trace()

                        os.system(f'ln -s {os.path.abspath(outputpath_pr)} {os.path.join(self.prefix,"best_images","output_pr_latest.png")}') 
                        """                      
                        os.system(f'ln -s {os.path.abspath(maskpath)} {os.path.join(self.prefix,"best_images","mask_pr_latest.png")}')                       
                        """
 

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
