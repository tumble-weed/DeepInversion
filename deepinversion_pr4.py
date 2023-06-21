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

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import colorful
from utils.utils import lr_cosine_policy, lr_policy, beta_policy, mom_cosine_policy, clip, denormalize, create_folder
import my_utils
import os

import sys
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
def set_optimizer(pr_models, lr1=3e-3, lr2=8e-4,lrexp=0.93,lr_s=0.03):
    sh_list = [name for name, params in pr_models[0].named_parameters() if 'sh' in name]
    sh_params = []
    other_params = []
    for pr_model in pr_models:
        sh_params.extend( list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in sh_list,
                            pr_model.named_parameters()))))
                         )
        other_params.extend( list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in sh_list,
                            pr_model.named_parameters()))))
        )
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
            self.start_noise = parameters["start_noise"]
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
        self.prefix = prefix
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
        N_PR = int(os.environ.get('N_PR',1))
        trends = defaultdict(list)
        ENABLE_PR = float(os.environ.get('ENABLE_PR',1))
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
                if ENABLE_PR == 0:
                    targets = targets[N_PR:]
                if ENABLE_PR == 1:
                    NVIEW = int(os.environ.get('NVIEW',1))
                    targets_pr = targets[:N_PR]
                    targets_other = targets[N_PR:]
                    targets = torch.cat([
                        targets_pr[:,None].repeat(1,NVIEW).flatten(),
                        targets_other
                    ],dim=0)
                    # import ipdb; ipdb.set_trace()


                    
        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        # PR ----------------------------------------
        inputs_other = torch.randn((self.bs - N_PR, 3, img_original, img_original), requires_grad=True, device=device,
                             dtype=data_type)
        #=======================================================================================
        do_clip = True
        from argparse import Namespace
        pr_args = Namespace()
        # pr_args.splatting_r = 0.015
        # pr_args.raster_n
        
        pr_args.splatting_r = 0.015
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
        
        pr_models = [CoreModel(pr_args,STANDARD=False,init_mode='random').to(device) for _ in range(N_PR)]
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
                pr_optimizer, pr_lr_scheduler = set_optimizer(pr_models, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
                optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
                # optimizer = optim.Adam(list(inputs_F_c.parameters()) + list(inputs_F_f.parameters()), lr=lr)
            elif self.setting_id == 1:
                #2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                pr_optimizer, pr_lr_scheduler = set_optimizer(pr_models, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
                optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True                                
            elif self.setting_id == 2:
                #20k normal resolution the closes to the paper experiments for ResNet50
                pr_optimizer, pr_lr_scheduler = set_optimizer(pr_models, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
                optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)                
                # optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)
                do_clip = True
                
            lr_scheduler_other = lr_cosine_policy(self.lr, 100, iterations_per_layer)                
            if False:
                if use_fp16:
                    static_loss_scale = 256
                    static_loss_scale = "dynamic"
                    _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)
            

            """
            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)
            """
            # PR ---------------------------------------------
            for pr_model in pr_models:
                pr_model.train()
            # END PR---------------------------------------------
            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                #============================================================================  
                from pytorch3d.ops import knn_points
                from point_radiance_modules.utils import remove_outlier              
                def repeat_pts(vertsparam,sh_param):
                    # import ipdb; ipdb.set_trace()
                    vertsparam.data = vertsparam.data.repeat(2,1)
                    sh_param.data = sh_param.data.repeat(2, 1)
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
                REFINE_AFTER = os.environ.get('REFINE_AFTER',None)
                REFINE_AFTER =  int(REFINE_AFTER) if REFINE_AFTER is not None else REFINE_AFTER
                REFINE_EVERY = os.environ.get('REFINE_EVERY',None)
                REFINE_EVERY = int(REFINE_EVERY) if REFINE_EVERY is not None else REFINE_EVERY
                if REFINE_AFTER is not None:
                    if iteration > REFINE_AFTER:
                        if (iteration%REFINE_EVERY) == 0:
                            # import ipdb; ipdb.set_trace()
                            for pr_model in pr_models:
                                pr_model.vertsparam.grad = None
                                pr_model.sh_param.grad = None
                                remove_out(pr_model.vertsparam,pr_model.sh_param)
                                repeat_pts(pr_model.vertsparam,pr_model.sh_param)
                                pr_model.vertsparam = torch.nn.Parameter(torch.tensor(pr_model.vertsparam))
                                pr_model.sh_param = torch.nn.Parameter(torch.tensor(pr_model.sh_param))
                                pr_model.train()
                            pr_optimizer, pr_lr_scheduler = set_optimizer(pr_models, lr1=pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
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
                    USE_FULL_SH_PARAMS = True
                    from pytorch3d.renderer import (
                        look_at_view_transform,
                        PerspectiveCameras,
                    )                      
                    from pytorch3d.renderer import cameras as p3dcameras
                    focal_length = 0.5
                    if os.environ.get('USE_FIXED_DIR',False) == '1':
                        # eye_at = (0,0,2)
                        # R, T = look_at_view_transform(eye = [eye_at],at=((0,0,0,),))
                        # elev_range = (-45,45)
                        # azim_range = (-90,90)
                        # # dist_range = (2,5)
                        # dist_range = (1,)                        
                        elev = 0
                        azim = 0
                        dist = 1
                        R, T = look_at_view_transform(dist=dist,azim=azim,elev=elev,at=((0,0,0,),))
                    else:
                        if False:
                            dist = 2
                            dirn = (np.random.random(3) - 0.5)*2
                            unit_dirn = dirn/np.linalg.norm(dirn)
                            eye_at = unit_dirn*dist
                            R, T = look_at_view_transform(eye = [eye_at],at=((0,0,0,),))
                            R,T = [R],[T]
                            NVIEW = 1                            
                        else:
                            elev_range = (-45,45)
                            if "ELEV_MAG" in os.environ:
                                elev_mag = int(os.environ['ELEV_MAG'])
                                elev_range = (-elev_mag,elev_mag)                            
                            azim_range = (-90,90)
                            if "AZIM_MAG" in os.environ:
                                azim_mag = int(os.environ['AZIM_MAG'])
                                azim_range = (-azim_mag,azim_mag)                            
                            """
                            if "DIST_MAG" in os.environ:
                                dist_mag = float(os.environ['DIST_MAG'])
                                dist_range = (1,dist_mag)                                                 
                                
                            """  
                            dist_min,dist_max = 1,2
                            if "DIST_MIN" in os.environ:
                                dist_min = float(os.environ['DIST_MIN'])
                            if "DIST_MAX" in os.environ:
                                dist_max = float(os.environ['DIST_MAX'])
                            dist_range = (dist_min,dist_max)
                            NVIEW = int(os.environ.get('NVIEW',1))         
                                                                                                                          
                            dist = dist_range[0] + (dist_range[1]-dist_range[0])*np.random.random(N_PR*NVIEW)
                            elev = elev_range[0] + (elev_range[1]-elev_range[0])*np.random.random(N_PR*NVIEW)
                            azim = azim_range[0] + (azim_range[1]-azim_range[0])*np.random.random(N_PR*NVIEW)
                            """
                            else:
                                elev = 0
                                azim = 0
                                dist = 2
                            import ipdb; ipdb.set_trace()
                            """
                            #==============================================================================================
                            R, T = look_at_view_transform(dist=dist,azim=azim,elev=elev,at=((0,0,0,),))
                            # import ipdb; ipdb.set_trace()
                            #==============================================================================================                            
                    inputs_from_pr = []
                    for pr_model in pr_models:
                        for ni in range(NVIEW):
                            Ri,Ti = R[ni],T[ni]
                            cameras = PerspectiveCameras(
                                    # focal_length=K[0][0] / K[0][2],
                                    focal_length = focal_length,
                                        device=device, 
                                        R=Ri, T=Ti)
                            Pmat = p3dcameras._get_sfm_calibration_matrix(cameras._N,cameras.device,cameras.focal_length,cameras.principal_point,orthographic=False)
                            K = Pmat[0,[0,1,3],:3]
                            target_pose = torch.zeros(4,4,device=device)
                            target_pose[:3, :3] = Ri
                            target_pose[:3, -1] = Ti
                            #--------------------------------------------------------------                        
                            if True:
                                if os.environ.get('ONLYBASE',False) == '1':
                                    pr_model.onlybase = True
                                    # print(colorful.red("using onlybase"))
                                inputs_from_pri = pr_model.forward_for_deepinversion(target_pose[None,...],K)
                                inputs_from_pr.append(inputs_from_pri)
                                """
                                my_utils.img_save2(inputs_from_pr,'inputs_from_pr.png',syncable=False)
                                import ipdb;ipdb.set_trace()
                                """
                            else:
                                poses = target_pose[None,...]
                                assert poses.ndim == 3
                                R, T = (poses[:, :3, :3]), poses[:, :3, -1]
                                # R, T = R, -(T[: ,None ,:] @ R)[: ,0]
                                """
                                PerspectiveCameras
                                PointsRasterizer
                                PointsRenderer
                                AlphaCompositor
                                Pointclouds
                                """
                                cameras = PerspectiveCameras(
                                    # focal_length=K[0][0] / K[0][2],
                                    focal_length=focal_length,
                                                            device=device, R=R, T=T)
                                rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pr_model.raster_settings)
                                renderer = PointsRenderer(
                                    rasterizer=rasterizer,
                                    compositor=AlphaCompositor()
                                )
                                point_cloud = Pointclouds(points=[pr_model.vertsparam], features=[pr_model.sh_param])
                                
                                if USE_FULL_SH_PARAMS:
                                    from point_radiance_modules.utils import get_rays
                                    from point_radiance_modules.sh import eval_sh
                                    #=================================================
                                    rays_o, rays_d = get_rays(pr_model.img_s, pr_model.img_s, torch.tensor(K).to(device), torch.tensor(poses[0],device='cuda'))
                                    rays_d = torch.nn.functional.normalize(rays_d, dim=2)   
                                    viewdir = (rays_d)    
                                    feat = renderer(point_cloud).flip(1) 
                                    base, shfeat = feat[..., :3], feat[..., 3:]
                                    shfeat = torch.stack(shfeat.split(3, 3), -1)
                                    image = base + eval_sh(pr_model.sh_n, shfeat, viewdir)
                                    # shfeat = torch.stack(shfeat.split(3, 3), -1)
                                    if pr_model.onlybase:
                                        image = base
                                    else:
                                        image = base + eval_sh(pr_model.sh_n, shfeat, viewdir)                                    
                                    inputs_from_pr = image       
                                    #=================================================
                                else:
                                    feat = renderer(point_cloud).flip(1)
                                    inputs_from_pr = feat[...,:3]       
                                my_utils.img_save2(inputs_from_pr,'inputs_from_pr.png',syncable=False)
                                            
                    inputs_from_pr = torch.cat(inputs_from_pr,dim=0)
                # import ipdb; ipdb.set_trace()
                #====================================================
                assert  inputs_from_pr.shape[1:] == (pr_args.img_s,pr_args.img_s,3)
                inputs_from_pr = inputs_from_pr.float()
                inputs_from_pr = inputs_from_pr.permute(0,3,1,2)
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
                
                if ENABLE_PR==0:
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
                if ENABLE_PR == 0:
                    main_loss_other = criterion(outputs, targets)
                    main_loss_pr = 0* inputs_from_pr.sum()

                else:
                    main_loss_pr = criterion(outputs[:NVIEW*N_PR], targets[:NVIEW*N_PR])
                    main_loss_other = criterion(outputs[NVIEW*N_PR:], targets[NVIEW*N_PR:])
                # END PR ------------------------------------------------------
                
                # R_prior losses
                # import ipdb; ipdb.set_trace()
                if ENABLE_PR:
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit[:NVIEW*N_PR])
                    loss_var_l1_other, loss_var_l2_other = get_image_prior_losses(inputs_jit[NVIEW*N_PR:])
                else:
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
                if ENABLE_PR:
                    loss_l2_other =0
                    if inputs_jit[NVIEW*N_PR:].shape[0] > 0:
                        loss_l2_other = torch.norm(inputs_jit[NVIEW*N_PR:].view(inputs_jit[NVIEW*N_PR:].shape[0], -1), dim=1).mean()
                    
                else:
                    # import ipdb; ipdb.set_trace()
                    loss_l2_other = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                # combining losses
                loss_aux = (
                            self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                            self.var_scale_l2 * loss_var_l2_other + \
                           self.var_scale_l1 * loss_var_l1_other + \
                           self.bn_reg_scale * loss_r_feature + \
                           self.l2_scale * loss_l2_other
                )

                if self.adi_scale!=0.0:
                    loss_aux += self.adi_scale * loss_verifier_cig
                trends['main_loss_pr'].append(main_loss_pr.item())
                trends['main_loss_other'].append(main_loss_other.item())
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
                        print("main criterion", criterion(outputs, targets).item())

                        if self.hook_for_display is not None:
                            self.hook_for_display(inputs, targets)

                # do image update
                if use_fp16:
                    # optimizer.backward(loss)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                pr_optimizer.step()
                optimizer_other.step()
                if do_clip:
                    if self.type_ == 'imagenet':
                        inputs_other.data = clip(inputs_other.data, use_fp16=use_fp16)                
                    elif self.type_ == 'facenet':
                        inputs_other.data = inputs_other.data.clip(-1,1)
                if False:
                    for g in optimizer.param_groups:
                        g["lr"] = lr * decay_rate ** (iteration / decay_steps)
                # clip color outlayers

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()
                print(colorful.orange(f"{self.prefix}"))
                if iteration % save_every==0 and (save_every > 0):
                    if local_rank==0:
                        # print('pickle save the images for running tests etc. later')
                        # import ipdb; ipdb.set_trace()
                        import pickle
                        with open('{}/best_images/output_{:05d}_gpu_{}.pkl'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank),'wb') as f:
                            pickle.dump(tensor_to_numpy(inputs),f)
                            
                        vutils.save_image(inputs,
                                          '{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank),
                                          normalize=True, scale_each=True, nrow=int(10))

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
