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
if False:
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
def set_optimizer(pr_model, lr1=3e-3, lr2=8e-4,lrexp=0.93,lr_s=0.03):
    sh_list = [name for name, params in pr_model.named_parameters() if 'sh' in name]
    sh_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in sh_list,
                            pr_model.named_parameters()))))
    other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in sh_list,
                            pr_model.named_parameters()))))
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
        
        

        self.net_teacher = net_teacher

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

        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

    def get_images(self, net_student=None, targets=None):

        trends = defaultdict(list)
        # view_errors = {'error':None,'azim':None,'n':None}
        view_errors = None
        N_AL_TRAIN = 10000
        ENABLE_PR = float(os.environ.get('ENABLE_PR',1))
        ENABLE_MESH = float(os.environ.get('ENABLE_MESH',0))
        print("get_images call")
        if torch.cuda.is_available():
            device = "cuda:0"
            device_other = "cuda:1"
        else:
            device = "cpu"

        net_teacher = self.net_teacher
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
        if targets is None:
            #only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor([random.randint(0, 999) for _ in range(self.bs)]).to(device)
            if not self.random_label:
                if os.environ.get('SINGLE_CLASS',False):
                    # dog class
                    targets = [int(os.environ['SINGLE_CLASS'])]
                else:
                    # import ipdb; ipdb.set_trace()
                    if self.type_ == 'imagenet':
                        # preselected classes, good for ResNet50v1.5
                        targets = [1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 277, 283, 292, 294, 309,
                                311,
                                325, 340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
                                967, 574, 487]
                    elif self.type_ == 'facenet':
                        targets = list(range(self.bs))
                if os.environ.get('PR_CLASS',None) is not None:
                    targets[0] = int(os.environ['PR_CLASS'])
                # import ipdb; ipdb.set_trace()
                targets = targets[:self.bs]
                targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to(device)
                if (ENABLE_PR == 0) and (ENABLE_MESH == 0):
                    targets = targets[1:]
                if (ENABLE_PR == 1) or (ENABLE_MESH == 1):
                    NVIEW = int(os.environ.get('NVIEW',1))
                    targets_pr = targets[:1]
                    targets_other = targets[1:]
                    targets = torch.cat([
                        targets_pr.repeat(NVIEW),
                        targets_other
                    ],dim=0)
                # import ipdb; ipdb.set_trace()


                    
        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        # PR ----------------------------------------
        inputs_other = torch.randn((self.bs - 1, 3, img_original, img_original), requires_grad=True, device=device,
                             dtype=data_type)
        #=======================================================================================
        do_clip = True
        from argparse import Namespace
        pr_args = Namespace()
        # pr_args.splatting_r = 0.015
        # pr_args.raster_n
        
        # pr_args.splatting_r = 0.015
        pr_args.splatting_r = 0.02
        pr_args.raster_n = 15
        pr_args.refine_n = 2
        pr_args.data_r = 0.012
        pr_args.step = 'brdf'
        pr_args.savemodel = None

        
        pr_args.lr1 = float(os.environ.get('LR1',22e-3))
        pr_args.lr2 = float(os.environ.get('LR2',8e-4))
        pr_args.lrexp = 0.93
        pr_args.lr_s = 0.03
        pr_args.img_s = img_original
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
        if ENABLE_PR:
            pr_model = CoreModel(pr_args,STANDARD=False,init_mode='random').to(device)
            pr_model.onlybase = True
        elif ENABLE_MESH:
            pr_model =  None
            from pytorch3d.utils import ico_sphere
            from pytorch3d.structures  import Meshes
            from pytorch3d.renderer import (
                TexturesVertex
            )            
            mesh_model = ico_sphere(5, device)
            # import ipdb; ipdb.set_trace()
            vert_offsets = torch.full(mesh_model.verts_packed().shape, 0.0, device=device, requires_grad=True)
            nvert = vert_offsets.shape[0]
            texture = torch.rand(nvert,3,device=device,requires_grad=True)
            textures_obj = TexturesVertex(verts_features=texture[None])
            # mesh_model.textures = textures_obj
            
            mesh_model  = Meshes(verts=[mesh_model.verts_packed()], faces=[mesh_model.faces_packed()],textures=textures_obj)
            # pr_optimizer = torch.optim.SGD([vert_offsets,texture], lr=1e-5, momentum=0.9)
            """
            pr_optimizer = torch.optim.Adam([
        {'params': vert_offsets, 'lr': 1e-3*1e-1*3},
        {'params': texture, 'lr': 0.1*1e-1}])
            """
            pr_optimizer = torch.optim.Adam([
        {'params': vert_offsets, 'lr': 1e-3*1e-1*3*1e-1},
        {'params': texture, 'lr': 0.1*1e-1*1e1}])            
            if False:
                from delaunay import render_mesh
                images = render_mesh(mesh_model,0.5)
            #=================================================
            # import ipdb; ipdb.set_trace()

            
        my_utils.cipdb('DBG_PR')
        # END PR ----------------------------------------
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
                
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000
                if self.setting_id == 2:
                    iterations_per_layer = 20000
                
            if os.environ.get('EPOCHS',None):
                iterations_per_layer = int(os.environ['EPOCHS'])
                # import ipdb; ipdb.set_trace()                    
            if view_errors is None:
                len_al_buffer = NVIEW*iterations_per_layer*(1 if skipfirst else 2)
                if N_AL_TRAIN is not None:
                    len_al_buffer = N_AL_TRAIN
                view_errors = {'error':np.zeros((len_al_buffer,)),
                               'azim':np.zeros((len_al_buffer,)),
                               'elev':np.zeros((len_al_buffer,)),
                               'dist':np.zeros((len_al_buffer,)),
                               'n':0
                               }
            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
            
            # print('check what the setting_id is and if what happens if you remove color clipping?')


            """
            :param setting_id: predefined settings for optimization:
                0 - will run low resolution optimization for 1k and then full resolution for 1k;
                1 - will run optimization on high resolution for 2k
                2 - will run optimization on high resolution for 20k            
            """            
            if self.setting_id == 0:
                if ENABLE_PR == 1:
                    pr_optimizer, pr_lr_scheduler = set_optimizer(pr_model, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
                optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
                # optimizer = optim.Adam(list(inputs_F_c.parameters()) + list(inputs_F_f.parameters()), lr=lr)
            elif self.setting_id == 1:
                if ENABLE_PR == 1:
                    #2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                    pr_optimizer, pr_lr_scheduler = set_optimizer(pr_model, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
                optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True                                
            elif self.setting_id == 2:
                if ENABLE_PR == 1:
                    #20k normal resolution the closes to the paper experiments for ResNet50
                    pr_optimizer, pr_lr_scheduler = set_optimizer(pr_model, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
                optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)                
                # optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)
                do_clip = True
                
            lr_scheduler_other = lr_cosine_policy(self.lr, 100, iterations_per_layer)                
            if use_fp16:
                net_teacher, _ = amp.initialize(net_teacher, [optimizer_other,pr_optimizer], opt_level="O2")                    
            if False:
                if use_fp16:
                    static_loss_scale = 256
                    static_loss_scale = "dynamic"
                    _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)
            

            """
            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)
            """
            # PR ---------------------------------------------
            if ENABLE_PR == 1:
                pr_model.train()
            # END PR---------------------------------------------
            running_dist_min = None
            running_dist_max = None
            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                #============================================================================  
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
                if ENABLE_PR == 1:
                    REFINE_AFTER = os.environ.get('REFINE_AFTER',None)
                    if REFINE_AFTER == '':
                        REFINE_AFTER = None
                    REFINE_AFTER =  int(REFINE_AFTER) if REFINE_AFTER is not None else REFINE_AFTER
                    REFINE_EVERY = os.environ.get('REFINE_EVERY',None)
                    if REFINE_EVERY == '':
                        REFINE_EVERY = None                
                    REFINE_EVERY = int(REFINE_EVERY) if REFINE_EVERY is not None else REFINE_EVERY
                    if REFINE_AFTER is not None:
                        if iteration > REFINE_AFTER:
                            if (iteration%REFINE_EVERY) == 0:
                                # import ipdb; ipdb.set_trace()
                                pr_model.vertsparam.grad = None
                                pr_model.sh_param.grad = None
                                original_n_pts = pr_model.vertsparam.shape[0]
                                remove_out(pr_model.vertsparam,pr_model.sh_param)
                                # import ipdb; ipdb.set_trace()
                                if os.environ.get('NO_REPEAT_PTS',None) != '1':
                                    
                                    repeat_pts(pr_model.vertsparam,pr_model.sh_param,target_n_vertices=original_n_pts)
                                assert pr_model.vertsparam.shape[0] == original_n_pts
                                pr_model.vertsparam = torch.nn.Parameter(torch.tensor(pr_model.vertsparam))
                                pr_model.sh_param = torch.nn.Parameter(torch.tensor(pr_model.sh_param))
                                pr_optimizer, pr_lr_scheduler = set_optimizer(pr_model, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
                                pr_model.train()
                trends['n_pts'].append(pr_model.vertsparam.shape[0])
                #============================================================================
                # learning rate scheduling
                lr_scheduler_other(optimizer_other, iteration_loc, iteration_loc)
                # PR ---------------------------------------------
                if True and 'visualize the image':
                    """
                    focal_length = 100
                    K = np.array([[focal_length,   0.       , focal_length/2.       ],
                                [  0.       , focal_length, focal_length/2.       ],
                                [  0.       ,   0.       ,   1.       ]])                                     
                    # K = np.array([[555.5555156,   0.       , 200.       ],
                    # [  0.       , 555.5555156, 200.       ],
                    # [  0.       ,   0.       ,   1.       ]])                                                           
                    # del globals()['camera_pose']
                    target_pose,elevation,azimuth = create_random_pose(
                    # Define the camera position and orientation parameters
                    elev_range = (82,82), # in degrees
                    azim_range = (180,180), # in degrees
                    """
                    #--------------------------------------------------------------
                    from view_sampling import sample_view_params
                    from pytorch3d.renderer import cameras as p3dcameras
                    '''
                    elev,azim,dist = sample_view_params(ENABLE_PR,
                       focal_length = 0.5,
                       USE_FULL_SH_PARAMS = True)
                    '''
                    focal_length = 0.5
                    elev,azim,dist = sample_view_params(
                      pr_model,
                      iteration_loc,
                       iteration,
                       trends,
                       device,
                       ENABLE_MESH,
                       N_AL_TRAIN,
                       view_errors,
                       )
                    #==============================================================================================
                    from pytorch3d.renderer import (
                        look_at_view_transform,
                        PerspectiveCameras,
                    )                      
                    # import ipdb; ipdb.set_trace()
                    R, T = look_at_view_transform(dist=dist,azim=azim,elev=elev,at=((0,0,0,),))
                    # import ipdb; ipdb.set_trace()
                    #==============================================================================================
                    from my_renderer import render
                    if ENABLE_PR:
                        inputs_from_pr,target_pose = render(
                            pr_model,R,T,focal_length,device,
                            ENABLE_PR,ENABLE_MESH,
                            pr_args=pr_args
                            )
                        
                        if True:
                            focal =focal_length
                            p1 = (pr_model.vertsparam for _ in range(NVIEW))
                            p1 = list(p1)
                            p1 = torch.stack(p1,dim=0)
                            dummy_texture = torch.ones_like(pr_model.sh_param[...,:3]).detach()
                            # feat1 = (dummy_texture for _ in range(NVIEW))
                            # feat1 = list(feat1)
                            feat1 = dummy_texture[None].repeat(NVIEW,1,1)
                            point_cloud_for_depth = Pointclouds(points=p1, features=feat1)
                            cameras = PerspectiveCameras(focal_length=focal,
                                        device=device, R=R, T=T)
                            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pr_model.raster_settings)
                            
                            if True:
                                from pytorch3d.renderer import (
                                    AlphaCompositor
                                )                                
                                renderer = PointsRenderer(
                                rasterizer=rasterizer,
                                compositor=AlphaCompositor())
                                masks_from_pr = renderer(point_cloud_for_depth).flip(1)
                                masks_from_pr = masks_from_pr.permute(0,3,1,2)
                                masks_from_pr = (masks_from_pr - masks_from_pr.min())/(masks_from_pr.max() - masks_from_pr.min())
                                
                            else:
                                for_depth = rasterizer(point_cloud_for_depth)
                                # zbuf = for_depth.zbuf.max(dim=-1)[0].unsqueeze(1)
                                # zbuf = for_depth.zbuf.min(dim=-1)[0].unsqueeze(1)
                                # https://github.com/facebookresearch/pytorch3d/blob/7aeedd17a4140eef139987e946a7017df7a97433/pytorch3d/renderer/points/rasterize_points.py#L75                                
                                zbuf = for_depth.zbuf[...,0].flip(1).unsqueeze(1)                    
                                # loss_var_l2_zbuf,_ = get_image_prior_losses(gaussian_blur(zbuf, 11, sigma=None))         
                                masks_from_pr = zbuf
                                
                                # import ipdb;ipdb.set_trace()
                                if False:
                                    max_z = masks_from_pr.max()
                                    min_z = masks_from_pr.min()
                                    
                                    # masks_from_pr[masks_from_pr==-1] = 0
                                    masks_from_pr = (masks_from_pr - min_z.detach())/(max_z - min_z).detach()
                                    # masks_from_pr = 1- masks_from_pr
                                else:
                                    """
                                    empty pixels are -1 => -1 == BAD
                                    nearer pixels are smaller => smaller == GOOD
                                    """
                                    max_z = masks_from_pr.max()
                                    masks_from_pr[masks_from_pr==-1] = max_z.detach()
                                    min_z = masks_from_pr.min()
                                    # masks_from_pr = (max_z.detach() - masks_from_pr)/(max_z - min_z).detach()
                                    """
                                    place max_z at 0 and min_z at 1
                                    """
                                    masks_from_pr = (max_z.detach() - masks_from_pr)/(max_z - min_z.detach()).detach()
                                    
                                    pass
                                assert masks_from_pr.min() >= 0
                                # import ipdb; ipdb.set_trace()                                       
                    elif ENABLE_MESH:
                        inputs_from_pr,target_pose = render(
                            None,R,T,focal_length,device,
                            ENABLE_PR,ENABLE_MESH,
                            mesh_model=mesh_model,vert_offsets=vert_offsets,pr_args=pr_args,
                            )
                        assert False,'copy from deepinversion_pr5'

                    # import ipdb; ipdb.set_trace()
                    #====================================================
                    assert  inputs_from_pr.shape[1:] == (pr_args.img_s,pr_args.img_s,3)
                    inputs_from_pr = inputs_from_pr.float()
                    inputs_from_pr = inputs_from_pr.permute(0,3,1,2)
                    inputs_from_pr0 = inputs_from_pr
                if self.type_ == 'facenet':
                    """
                    def fixed_image_standardization(image_tensor):
                        processed_tensor = (image_tensor - 127.5) / 128.0
                        return processed_tensor        
                    """                                    
                    inputs_from_pr = (inputs_from_pr - 0.5)*2
                elif self.type_ == 'imagenet':
                    from utils.utils import normalize
                    inputs_from_pr = normalize(inputs_from_pr,inplace=False)
                # inputs_from_pr = inputs_from_pr * float(os.environ.get('ENABLE_PR',1))
                print(colorful.red("see shape of inputs_from_pr"))
                # import ipdb; ipdb.set_trace()

                # END PR---------------------------------------------
                # END PR---------------------------------------------
                # import ipdb; ipdb.set_trace()
                
                if ENABLE_PR==0 and ENABLE_MESH == 0:
                    inputs = torch.cat([inputs_other],dim=0)
                else:
                    inputs = torch.cat([inputs_from_pr,inputs_other],dim=0)
                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                    # inputs_jit_other = pooling_function(inputs_other)
                else:
                    inputs_jit = inputs
                    # inputs_jit_other = (inputs_other)
                #=============================================================
                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                pr_optimizer.zero_grad()
                optimizer_other.zero_grad()
                net_teacher.zero_grad()

                outputs = net_teacher(inputs_jit)


                outputs = self.network_output_function(outputs)

                # R_cross classification loss
                # print('check if targets are just indicators')
                # import ipdb; ipdb.set_trace()


                # PR --------------------------------------------------------
                # calculate loss for the PR as well as non PR
                # loss = criterion(outputs, targets)
                
                if (ENABLE_PR == 0) and (ENABLE_MESH == 0):
                    main_loss_other = criterion(outputs, targets).mean()
                    main_loss_pr = 0* inputs_from_pr.sum()
                else:
                    # import ipdb; ipdb.set_trace()
                    per_sample_main_loss_pr = criterion(outputs[:NVIEW], targets[:NVIEW])
                    main_loss_pr = per_sample_main_loss_pr.mean()
                    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                    # assert view_errors['n'] + NVIEW <= view_errors['error'].shape[0]
                    
                    # np.concatenate([view_errors['error'],tensor_to_numpy(per_sample_main_loss_pr)],axis=0)
                    if self.type_ != 'facenet':
                        if (N_AL_TRAIN is not None) and view_errors['n'] == N_AL_TRAIN:
                            error = tensor_to_numpy(per_sample_main_loss_pr)
                            view_errors['azim'][:- azim.shape[0]] = view_errors['azim'][azim.shape[0]:]
                            view_errors['elev'][:- elev.shape[0]] = view_errors['elev'][elev.shape[0]:]
                            view_errors['dist'][:- dist.shape[0]] = view_errors['dist'][dist.shape[0]:]
                            view_errors['error'][:- error.shape[0]] = view_errors['error'][error.shape[0]:]
                            view_errors['azim'][- azim.shape[0]:] = azim
                            view_errors['elev'][- elev.shape[0]:] = elev
                            view_errors['dist'][- dist.shape[0]:] = dist
                            view_errors['error'][- error.shape[0]:] = error
                            view_errors['n'] = min(N_AL_TRAIN,view_errors['n']+NVIEW)
                        else:
                            view_errors['error'][view_errors['n']:view_errors['n']+NVIEW] = tensor_to_numpy(per_sample_main_loss_pr)
                            view_errors['azim'][view_errors['n']:view_errors['n']+NVIEW] =  azim
                            view_errors['elev'][view_errors['n']:view_errors['n']+NVIEW] =  elev
                            view_errors['dist'][view_errors['n']:view_errors['n']+NVIEW] =  dist
                            view_errors['n'] = view_errors['n']+NVIEW                    

                    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                    main_loss_other = criterion(outputs[NVIEW:], targets[NVIEW:]).mean()
                    # import ipdb; ipdb.set_trace()
                    pr_acc = (outputs[:NVIEW].max(dim=1)[0] ==  outputs[range(NVIEW),targets[:NVIEW]]).float().mean()
                    
                # END PR ------------------------------------------------------
                
                # R_prior losses
                # import ipdb; ipdb.set_trace()
                if ENABLE_PR:
                    # loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit[:NVIEW])
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(gaussian_blur(inputs_jit[:NVIEW], 31, sigma=None))

                    loss_var_l1_first, loss_var_l2_first = get_image_prior_losses(inputs_jit[:1])
                    loss_var_l1_other, loss_var_l2_other = get_image_prior_losses(inputs_jit[NVIEW:])
                    
                    trends['loss_var_l2'].append(loss_var_l2.item())
                    trends['loss_var_l2_first'].append(loss_var_l2_first.item())
                    if False:
                        focal =focal_length
                        p1 = (pr_model.vertsparam for _ in range(10))
                        p1 = list(p1)
                        feat1 = (pr_model.sh_param for _ in range(10))
                        feat1 = list(feat1)
                        point_cloud_for_depth = Pointclouds(points=p1, features=feat1)
                        cameras = PerspectiveCameras(focal_length=focal,
                                    device=device, R=R[:10], T=T[:10])
                        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pr_model.raster_settings)
                        for_depth = rasterizer(point_cloud_for_depth)
                        # zbuf = for_depth.zbuf.max(dim=-1)[0].unsqueeze(1)
                        # zbuf = for_depth.zbuf.min(dim=-1)[0].unsqueeze(1)
                        # https://github.com/facebookresearch/pytorch3d/blob/7aeedd17a4140eef139987e946a7017df7a97433/pytorch3d/renderer/points/rasterize_points.py#L75
                        zbuf = for_depth.zbuf[...,0].flip(1).unsqueeze(1)                    
                    loss_var_l2_zbuf,_ = get_image_prior_losses(masks_from_pr)
                    trends['loss_var_l2_zbuf'].append(loss_var_l2_zbuf.item())
                else:
                    assert False
                    loss_var_l1_other, loss_var_l2_other = get_image_prior_losses(inputs_jit)
                    loss_var_l1, loss_var_l2 = 0,0

                # print('see step by step how the batch norm loss is calculated')
                # import ipdb; ipdb.set_trace()
                # R_feature loss
                
                # rescale = [self.first_bn_multiplier] + [ (0.1 if os.environ.get('SINGLE_CLASS',False) else 1) for _ in range(len(self.loss_r_feature_layers)-1)]
                rescale = [self.first_bn_multiplier] + [ (1) for _ in range(len(self.loss_r_feature_layers)-1)]

                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])

                # R_ADI
                loss_verifier_cig = torch.zeros(1)
                if self.adi_scale!=0.0:
                    if self.detach_student:
                        outputs_student = net_student(inputs_jit).detach()
                    else:
                        outputs_student = net_student(inputs_jit)

                    T = 3.0
                    if 1:

                        T = 3.0
                        # Jensen Shanon divergence:
                        # another way to force KL between negative probabilities
                        P = nn.functional.softmax(outputs_student / T, dim=1)
                        Q = nn.functional.softmax(outputs / T, dim=1)
                        M = 0.5 * (P + Q)

                        P = torch.clamp(P, 0.01, 0.99)
                        Q = torch.clamp(Q, 0.01, 0.99)
                        M = torch.clamp(M, 0.01, 0.99)
                        eps = 0.0
                        loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                         # JS criteria - 0 means full correlation, 1 - means completely different
                        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                    if local_rank==0:
                        if iteration % save_every==0:
                            print('loss_verifier_cig', loss_verifier_cig.item())

                # l2 loss on images
                # loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()
                if ENABLE_PR == 1 or ENABLE_MESH == 1:
                    loss_l2_other = 0
                    if inputs_other.shape[0]>0:
                        loss_l2_other = torch.norm(inputs_jit[NVIEW:].view(inputs_jit[NVIEW:].shape[0], -1), dim=1).mean()
                    if True:
                        loss_l2_pr = torch.norm(inputs_from_pr.reshape(inputs_from_pr.shape[0], -1), dim=1).mean()
                        trends['loss_l2_pr'].append(loss_l2_pr.item())
                        loss_l2_masks = 0.01*torch.norm(masks_from_pr.reshape(masks_from_pr.shape[0], -1), dim=1).mean()
                        # loss_l2_masks = torch.abs(masks_from_pr.reshape(masks_from_pr.shape[0], -1)).sum(dim=1).mean()
                        trends['loss_l2_masks'].append(loss_l2_masks.item())                    
                    else:
                        """
                        # inputs_jit_pr = inputs_jit[:NVIEW]
                        loss_l2_pr = torch.norm((inputs_from_pr*(inputs_from_pr.abs()>1).float() ).view(inputs_from_pr.shape[0], -1), dim=1).mean()
                        """
                else:
                    # import ipdb; ipdb.set_trace()
                    loss_l2_other = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()
                    
                L2_PR = float(os.environ.get('L2_PR', 0))
                if ENABLE_PR==1:
                    from delaunay import get_mesh,get_mesh_volume
                    vertsparam_for_volume = pr_model.vertsparam.clone()
                    vertsparam_for_volume.retain_grad()
                    mesh = get_mesh(vertsparam_for_volume)
                    mesh_volume = get_mesh_volume(mesh)
                    trends['mesh_volume'].append(mesh_volume.item())
                    MESH_VOLUME = float(os.environ.get('MESH_VOLUME', 0))
                    # import ipdb; ipdb.set_trace()
                    vertsparam_for_l2_vertsparam = pr_model.vertsparam.clone()            
                    vertsparam_for_l2_vertsparam.retain_grad()  
                    loss_l2_vertsparam = (vertsparam_for_l2_vertsparam - vertsparam_for_l2_vertsparam.mean(dim=0,keepdim=True)).norm(dim=1).sum()                      
                    # loss_l2_vertsparam = pr_model.vertsparam.norm(dim=1).sum()
                    trends['loss_l2_vertsparam'].append(loss_l2_vertsparam.item())
                    L2_VERTS = float(os.environ.get('L2_VERTS', 0))

                    # hack(locals(),L2_VERTS = 0)
                    from smoothness_3d import total_variation_3d_loss
                    tv_3d = total_variation_3d_loss(pr_model.vertsparam[None], pr_model.sh_param[None], k = 6)
                    trends['tv_3d'].append(tv_3d.item())
                    
                # combining losses
                
                    EXTRA_L2_MULTIPLIER = 1
                    # hack(locals(),EXTRA_L2_MULTIPLIER=100)

                    loss_aux = (
                            # EXTRA_L2_MULTIPLIER * self.var_scale_l2 * loss_var_l2 + \
                            # self.var_scale_l1 * loss_var_l1 + \
                              (10**(float(targets[0]==933)*-3) ) *1e1*loss_l2_masks +\
                                1*1e2 * loss_var_l2_zbuf +                                 
                            1*self.var_scale_l2 * loss_var_l2_other + \
                            self.var_scale_l1 * loss_var_l1_other + \
                            self.bn_reg_scale * loss_r_feature + \
                            1*self.l2_scale * loss_l2_other +\
                            # L2_PR* self.l2_scale * loss_l2_pr +\
                            0*L2_VERTS * loss_l2_vertsparam +\
                            # MESH_VOLUME * mesh_volume +\
                            0*1e-4* tv_3d
                    )
                elif ENABLE_MESH == 1:
                    assert False
                    assert False,'copy stuff from pr5'
                    from pytorch3d.loss import (
                        chamfer_distance, 
                        mesh_edge_loss, 
                        mesh_laplacian_smoothing, 
                        mesh_normal_consistency,
                    )
                    w_edge = 1.0 * 10
                    # Weight for mesh normal consistency
                    w_normal = 0.01 * 10
                    # Weight for mesh laplacian smoothing
                    w_laplacian = 0.1  * 10                  
                    # import ipdb; ipdb.set_trace()
                    loss_edge = mesh_edge_loss(shifted_mesh_model)
                    loss_normal = mesh_normal_consistency(shifted_mesh_model)
                    loss_laplacian = mesh_laplacian_smoothing(shifted_mesh_model, method="uniform")
                    loss_aux = (
                            self.var_scale_l2 * loss_var_l2 + \
                            self.var_scale_l1 * loss_var_l1 + \
                            
                            self.var_scale_l2 * loss_var_l2_other + \
                            self.var_scale_l1 * loss_var_l1_other + \
                            self.bn_reg_scale * loss_r_feature + \
                            self.l2_scale * loss_l2_other +\
                            L2_PR* self.l2_scale * loss_l2_pr +\
                            loss_edge * w_edge +\
                            loss_normal * w_normal +\
                            loss_laplacian * w_laplacian 
                    )
                    
                    
                else:
                    assert False

                if self.adi_scale!=0.0:
                    loss_aux += self.adi_scale * loss_verifier_cig
                trends['main_loss_pr'].append(main_loss_pr.item())
                trends['main_loss_other'].append(main_loss_other.item())
                if len(trends['pr_acc']) and  max(trends['pr_acc']) == 1 and pr_acc < 0.1:
                    for lname in ['pr_acc','main_loss_pr','main_loss_other','mesh_volume','n_pts','dist_min','loss_var_l2','loss_l2_vertsparam','loss_l2_pr','loss_var_l2_zbuf','loss_var_l2_first','tv_3d']:
                        my_utils.save_plot(trends[lname],lname,f'{os.path.join(self.prefix,"best_images",lname)}.png')                    
                    def run_for_arbitrary_pose():
                        elev1,azim1,dist1 = sample_view_params(
                                            pr_model,
                                            iteration_loc,
                                            iteration,
                                            trends,
                                            device,
                                            ENABLE_MESH,
                                            N_AL_TRAIN,
                                            view_errors,
                                            )
                        R1, T1 = look_at_view_transform(dist=dist1,azim=azim1,elev=elev1,at=((0,0,0,),))
                        inputs_from_pr1,target_pose1 = render(
                                                    pr_model,R1,T1,focal_length,device,
                                                    ENABLE_PR,ENABLE_MESH,
                                                    pr_args=pr_args
                                                    )
                        inputs_from_pr1 = inputs_from_pr1.float()
                        inputs_from_pr1 = inputs_from_pr1.permute(0,3,1,2)
                        inputs_from_pr1 = normalize(inputs_from_pr1,inplace=False)
                        outputs1 = net_teacher(inputs_from_pr1)
                        pr_acc1 = (outputs1[:NVIEW].max(dim=1)[0] ==  outputs1[range(NVIEW),targets[:NVIEW]]).float().mean()
                        print(pr_acc1)          
                    run_for_arbitrary_pose()
                    import ipdb; ipdb.set_trace()
                trends['pr_acc'].append(pr_acc.item())
                """
                next? actual classification oss?
                """                
                loss = (
                    #1e-3
                    float(os.environ.get('PR_LOSS',1e0))* self.main_loss_multiplier * main_loss_pr + 
                    self.main_loss_multiplier * main_loss_other + 
                    loss_aux
                )

                if local_rank==0:
                    if iteration % save_every==0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print("main criterion", criterion(outputs, targets).mean().item())

                        if self.hook_for_display is not None:
                            self.hook_for_display(inputs, targets)

                # do image update
                if use_fp16:
                    # optimizer.backward(loss)
                    with amp.scale_loss(loss, [optimizer_other,pr_optimizer]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # import ipdb; ipdb.set_trace()
                pr_optimizer.step()
                optimizer_other.step()
                
                # import ipdb; ipdb.set_trace()
                """
                if iteration > 100:
                    if (pr_model.vertsparam[:,:3].abs() > 1).any():
                        print(pr_model.vertsparam[:,:3].abs().max())
                        import ipdb; ipdb.set_trace()
                """
                """
                if iteration > 500:
                    # my_utils.cipdb('DBG_HIGH_SH')
                    import ipdb; ipdb.set_trace()
                """
                if do_clip:
                    if self.type_ == 'imagenet':
                        
                        inputs_other.data = clip(inputs_other.data, use_fp16=use_fp16)                
                        if ENABLE_PR == 1:
                            sh_param = pr_model.sh_param.detach().clone()
                            pr_model.sh_param.data.copy_(
                                sh_param.data.clamp(0,1)
                            )
                        elif ENABLE_MESH == 1:
                            # texture = pr_model.sh_param.detach().clone()
                            texture.data.copy_(
                                texture.data.clamp(0,1)
                            )
                            mesh_model.textures = TexturesVertex(verts_features=texture[None])
                    elif self.type_ == 'facenet':
                        # assert False
                        if ENABLE_PR == 1:
                            inputs_other.data = inputs_other.data.clip(-1,1)
                            # import ipdb; ipdb.set_trace()
                            sh_param = pr_model.sh_param.detach().clone()
                            pr_model.sh_param.data.copy_(
                                sh_param.data.clamp(0,1)
                            )
                        elif ENABLE_MESH == 1:
                            assert False
                    if ENABLE_PR == 1 and True:
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
                if False:
                    for g in optimizer.param_groups:
                        g["lr"] = lr * decay_rate ** (iteration / decay_steps)
                # clip color outlayers

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()
                print(colorful.orange(f"{self.prefix}"))
                print(colorful.orange(f"{PDB_PORT}"))
                if (save_every > 0)  and any([iteration % save_every==0  , iteration in [10,20,30,40,50,60,70,80,90]]):
                    if local_rank==0:
                        # print('pickle save the images for running tests etc. later')
                        # import ipdb; ipdb.set_trace()
                        if False:
                            import pickle
                            with open('{}/best_images/output_{:05d}_gpu_{}.pkl'.format(self.prefix,
                                                                                            iteration // save_every,
                                                                                            local_rank),'wb') as f:
                                pickle.dump(tensor_to_numpy(inputs),f)
                        for lname in ['pr_acc','main_loss_pr','main_loss_other','mesh_volume','n_pts','dist_min','loss_var_l2','loss_l2_vertsparam','loss_l2_pr','loss_var_l2_zbuf','loss_var_l2_first','tv_3d','loss_l2_masks']:
                            my_utils.save_plot(trends[lname],lname,f'{os.path.join(self.prefix,"best_images",lname)}.png')
                        outputpath = '{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank)
                        vutils.save_image(inputs_from_pr0,
                                          outputpath,
                                          normalize=False, scale_each=False, nrow=int(10))
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","output_latest.png")}')
                        # import ipdb; ipdb.set_trace()
                        os.system(f'ln -s {os.path.abspath(outputpath)} {os.path.join(self.prefix,"best_images","output_latest.png")}')
                        if ENABLE_PR==1 and not os.environ.get('IGNORE_VIZ_ZBUF',False) == '1':
                            # focal = K[0,0]
                            focal =focal_length
                            point_cloud = Pointclouds(points=[pr_model.vertsparam], features=[pr_model.sh_param])
                            cameras = PerspectiveCameras(focal_length=focal,
                                     device=device, R=target_pose[:3, :3].unsqueeze(0), T=target_pose[:3, -1].unsqueeze(0))
                            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pr_model.raster_settings)
                            for_depth = rasterizer(point_cloud)
                            # zbuf = for_depth.zbuf.max(dim=-1)[0].unsqueeze(1)
                            # zbuf = for_depth.zbuf.min(dim=-1)[0].unsqueeze(1)
                            # https://github.com/facebookresearch/pytorch3d/blob/7aeedd17a4140eef139987e946a7017df7a97433/pytorch3d/renderer/points/rasterize_points.py#L75
                            zbuf = for_depth.zbuf[...,0].flip(1)
                            zbuf2 = (zbuf - zbuf.min())/(zbuf.max() - zbuf.min())
                            zbuf2 = 1 - zbuf2
                            zbuf2[zbuf==-1] = 0
                            zbuf_savename = f'{self.prefix}/best_images/zbuf_{iteration // save_every}.png'
                            my_utils.img_save(tensor_to_numpy(zbuf2)[0],zbuf_savename
                                              )
                            os.system(f'unlink {os.path.join(self.prefix,"best_images","zbuf_latest.png")}')
                            os.system(f'ln -s {os.path.abspath(zbuf_savename)} {os.path.join(self.prefix,"best_images","zbuf_latest.png")}')
                            # import ipdb; ipdb.set_trace()
                        # import ipdb; ipdb.set_trace()
                        if ENABLE_PR == 1:
                            from delaunay import render_as_mesh
                            rendered_mesh,mesh = render_as_mesh(pr_model.vertsparam.detach(),focal = focal_length)
                            my_utils.img_save(tensor_to_numpy(rendered_mesh)[0],
                                              f'{self.prefix}/best_images/rendered_mesh_{iteration // save_every}.png')
                        
                        if iteration == save_every:
                            
                            if os.system(f'ln -s {self.prefix} {self.prefix_alias}') != 0:
                                os.system(f'rm {self.prefix_alias}')
                                
                                os.system(f'ln -s {os.path.abspath(self.prefix)} {self.prefix_alias}')

        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        pr_optimizer.state = collections.defaultdict(dict)
        optimizer_other.state = collections.defaultdict(dict)

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
