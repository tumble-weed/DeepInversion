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
FIXTHIS = False
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


def normalize_for_cnn(inputs_from_pr,type_):
    if type_ == 'facenet':
        """
        def fixed_image_standardization(image_tensor):
            processed_tensor = (image_tensor - 127.5) / 128.0
            return processed_tensor        
        """                                    
        inputs_from_pr = (inputs_from_pr - 0.5)*2
    elif type_ == 'imagenet':
        from utils.utils import normalize
        inputs_from_pr = normalize(inputs_from_pr,inplace=False)
    return inputs_from_pr
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
        
        trends = defaultdict(list)
        # view_errors = {'error':None,'azim':None,'n':None}
        view_errors = None
        N_AL_TRAIN = 10000
        ENABLE_PR = float(os.environ.get('ENABLE_PR',1))
        ENABLE_MESH = float(os.environ.get('ENABLE_MESH',0))
        USE_TRANSPARENCY = os.environ.get('USE_TRANSPARENCY',False) == '1'
        print("get_images call")
        if torch.cuda.is_available():
            device = "cuda:0"
            device_other = "cuda:1"
        else:
            device = "cpu"
        os.environ['DGCNN'] = '1'
        print(colorful.yellow_on_red('setting DGNN to True'))
        if os.environ.get('DGCNN') == '1':
            from modelnet_utils import get_dgcnn
            modelnet_model = get_dgcnn()

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
                targets = [targets[0] for _ in targets + [None]]
                # import ipdb; ipdb.set_trace()
                targets = targets[:self.bs]
                targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to(device)
                if (ENABLE_PR == 0) and (ENABLE_MESH == 0):
                    targets = targets[1:]
                NVIEW = int(os.environ.get('NVIEW',1))

        if os.environ.get('DGCNN',False) == '1':
            # 'lamp' (confirm this is class 19)
            modelnet_targets = torch.tensor([19 ],device=device)
            # imagenet 'table lamp'
            targets=  torch.tensor([846 for _ in range(self.bs)],device=device)
            # optimizer_modelnet = torch.optim.Adam(
            #     *,lr= *
            # )
        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        # PR ----------------------------------------
        inputs_other = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device=device,
                             dtype=data_type)
        #=======================================================================================
        
        # if not USE_TRANSPARENCY:
        #     iipd    
        # utils.cipdb('')
        # USE_TRANSPARENCY = True
        if USE_TRANSPARENCY:
            print(colorful.green("using transparency"))
        alphas  = None




            
        my_utils.cipdb('DBG_PR')
        # END PR ----------------------------------------
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        if False:
            from view_errors import ViewErrors
            view_errors_obj = ViewErrors()
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
            other_param = pr_model.vertsparam       

            if USE_TRANSPARENCY:

            # other_param = []
            # alphas.requires_grad_(False)
            standard_inverter_obj(self.setting_id)

            # if USE_TRANSPARENCY:
            #     import ipdb; ipdb.set_trace()
                

            if use_fp16:
                net_teacher, _ = amp.initialize(net_teacher, [optimizer_other,pr_optimizer], opt_level="O2")                    

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
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

                    from my_renderer import render
                    #TODO: get masks from the renderer for PR or MESH                                
                    #====================================================
                    assert  inputs_from_pr.shape[1:] == (pr_args.img_s,pr_args.img_s,3)
                    inputs_from_pr = inputs_from_pr.float()
                    inputs_from_pr = inputs_from_pr.permute(0,3,1,2)
                    inputs_from_pr_pre_norm = inputs_from_pr
                    
                inputs_from_pr = normalize_for_cnn(inputs_from_pr_pre_norm,self.type_)

                
                # inputs_from_pr = inputs_from_pr * float(os.environ.get('ENABLE_PR',1))
                print(colorful.red("see shape of inputs_from_pr"))
                # import ipdb; ipdb.set_trace()
                assert surface_z.shape[1] == 1

                inputs = inputs_other
                # perform downsampling if needed
                from distillation import create_input_for_distillation
                inputs, inputs_for_prior = create_input_for_distillation(masks_from_pr,zbuf)



                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                    # inputs_jit_other = pooling_function(inputs_other)
                    inputs_for_prior_jit = pooling_function(inputs_for_prior)
                else:
                    inputs_jit = inputs
                    inputs_for_prior_jit = inputs_for_prior
                    # inputs_jit_other = (inputs_other)
                #=============================================================
                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
                inputs_for_prior_jit = torch.roll(inputs_for_prior_jit, shifts=(off1, off2), dims=(2, 3))
                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))
                    inputs_for_prior_jit = torch.flip(inputs_for_prior_jit, dims=(3,))

                # forward pass
                """
                pr_optimizer.zero_grad()
                optimizer_other.zero_grad()
                """
                for inverter_obj in inverter_objs:
                    inverter_obj.zero_grad()
                net_teacher.zero_grad()

                outputs = net_teacher(inputs_jit)


                outputs = self.network_output_function(outputs)


                # PR --------------------------------------------------------
                # calculate loss for the PR as well as non PR
                # loss = criterion(outputs, targets)
                
                main_loss_other(outputs)
                main_loss_pr(inputs_from_pr)
                inverter_obj.get_loss()
                                    
                # END PR ------------------------------------------------------
                if True:
                    from losses import get_verifier_loss
                    from losses import get_bn_loss
                    
                    loss_r_feature = get_bn_loss(net_student,self.loss_r_feature_layers,self.first_bn_multiplier)

                    # R_ADI
                    loss_verifier_cig = torch.zeros(1)
                    if self.adi_scale!=0.0:
                        loss_verifier_cig = get_verifier_loss(inputs_jit,output,net_student,
                                            self.adi_scale,self.detach_student)        
                        if local_rank==0:
                            if iteration % save_every==0:
                                print('loss_verifier_cig', loss_verifier_cig.item())

                if os.environ.get('DGCNN',False) == '1':
                    """
                    sample points from point cloud
                    """
                    """
                    import pytorch3d.ops
                    self.bs_modelnet = 1
                    os.environ['DBG_FIXED_SAMPLED_POINTS'] = '1'
                    
                    print(colorful.yellow_on_red(
                        'DBG_FIXED_SAMPLED_POINTS' + os.environ.get("DBG_FIXED_SAMPLED_POINTS",False)
                        )
                        )
                    sampled_vertsparam,ix_of_sampled_vertsparam = pytorch3d.ops.sample_farthest_points(    
                        torch.cat([pr_model.vertsparam.unsqueeze(0) for _ in range(self.bs_modelnet)],dim=0), 
                        lengths= None, K = 1024, 
                        random_start_point = (True if not os.environ.get('DBG_FIXED_SAMPLED_POINTS',False) else False)
                        )
                    sampled_vertsparam = sampled_vertsparam.permute(0,2,1)
                    """
                    sampled_vertsparam = dgcnn_inverter_obj.forward()
                    modelnet_scores = modelnet_model(sampled_vertsparam)
                    modelnet_probs = torch.softmax(modelnet_scores,dim=-1)
                    loss_modelnet = -modelnet_scores[:,modelnet_targets]
                    loss_modelnet = criterion(modelnet_scores,modelnet_targets.repeat(self.bs_modelnet)).sum()
                    # import ipdb;ipd.set_trace()
                    pass
                # l2 loss on images
                # loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()
                from distillation import get_distillation_losses
                loss_l2_other,loss_l2_pr,loss_l2_masks = get_distillation_losses(inputs_for_prior_jit,
                            inputs_from_pr,
                            masks_from_pr,
                            ENABLE_PR,
                            ENABLE_MESH,
                            trends) 

                    
                L2_PR = float(os.environ.get('L2_PR', 0))
                if ENABLE_PR==1:
                    mesh_volume,loss_l2_vertsparam,tv_3d = get_3d_losses(pr_model,trends)
                    if 'surface_z':
                        loss_l2_surface_z_coeff = 0
                        loss_var_l2_surface_z_coeff = 0
                        if USE_TRANSPARENCY:
                            # surface_z0 = surface_z
                            # surface_z1 = surface_z0.permute(0,3,1,2)
                            surface_z1 = surface_z
                            # loss_l2_surface_z =  ((zbuf_rand)**2).sum()
                            loss_l2_surface_z = (surface_z1**2).sum()
                            trends['loss_l2_surface_z'].append(loss_l2_surface_z.item())
                            _,loss_var_l2_surface_z = get_image_prior_losses(gaussian_blur(surface_z1, 11, sigma=None))
                            trends['loss_var_l2_surface_z'].append(loss_var_l2_surface_z.item())
                    # trends['loss_var_l2_masks'].append(loss_var_l2_masks.item())
                    
                    EXTRA_L2_MULTIPLIER = 1
                    # hack(locals(),EXTRA_L2_MULTIPLIER=100)
                    # hack(locals(),L2_VERTS=0.001)
                    # hack(locals(),L2_VERTS=1000)
                    L2_VERTS = 10
                    DISTILL_AFTER = 0
                    DISABLE = 0
                    ENABLE = 1
                    DBG_OVAL=False
                    # DBG_REDO_MASK = False
                    DBG_REDO_MASK_SMOOTH = False
                    DBG_NO_BN = False
                    #========================================================================
                    loss_l2_masks_coeff = DISABLE* 0.01*1*(10**(float(targets[0]==933)*2) ) *1e1
                    loss_var_l2_masks_coeff = 1e2*max(DISABLE,1 if DBG_REDO_MASK_SMOOTH else 0)*1*1e-1 
                    loss_var_l2_zbuf_coeff = (DISABLE if DBG_OVAL else 1)*ENABLE*1e-1
                    loss_l2_zbuf_coeff = DISABLE*dict(one = 1,lowest = 1e-6,between=1e-3)['lowest']
                    loss_l2_vertsparam_coeff = (1e2 if DBG_OVAL else 1e-1)*ENABLE*1e-5*float(iteration_loc >= DISTILL_AFTER) *L2_VERTS
                    tv_3d_coeff = DISABLE*float(iteration_loc >= DISTILL_AFTER) *1e-4
                    if USE_TRANSPARENCY:
                        loss_l2_surface_z_coeff = 1.
                        loss_var_l2_surface_z_coeff = 1.               
                        loss_l2_masks_coeff = DISABLE
                        loss_var_l2_masks_coeff = DISABLE
                        loss_var_l2_zbuf_coeff = DISABLE
                        loss_l2_zbuf_coeff = DISABLE
                        loss_l2_vertsparam_coeff = DISABLE
                        tv_3d_coeff = DISABLE
                    #========================================================================
                    loss_aux = (
                            # float(iteration_loc > DISTILL_AFTER) *EXTRA_L2_MULTIPLIER * self.var_scale_l2 * loss_var_l2 + \
                            # float(iteration_loc > DISTILL_AFTER) *self.var_scale_l1 * loss_var_l1 + \
                            loss_l2_masks_coeff*loss_l2_masks +\
                            loss_var_l2_masks_coeff * loss_var_l2_masks + 
                            loss_var_l2_zbuf_coeff*loss_var_l2_zbuf +
                            loss_l2_zbuf_coeff*loss_l2_zbuf+
                            DISABLE*1*self.var_scale_l2 * loss_var_l2_other + \
                            DISABLE*self.var_scale_l1 * loss_var_l1_other + \
                            (DISABLE if DBG_NO_BN else ENABLE)*self.bn_reg_scale * loss_r_feature + \
                            1*self.l2_scale * loss_l2_other +\
                            # L2_PR* self.l2_scale * loss_l2_pr +\
                            loss_l2_vertsparam_coeff * loss_l2_vertsparam +\
                            # MESH_VOLUME * mesh_volume +\
                            tv_3d_coeff* tv_3d +
                            # USE_TRANSPARENCY
                            (
                                loss_l2_surface_z_coeff * loss_l2_surface_z +\
                                loss_var_l2_surface_z_coeff * loss_var_l2_surface_z
                            )
                    )

                    DBG_LOSS = False
                    if DBG_LOSS and (iteration_loc > 800):
                        print(colorful.red("stopping after 800"))
                        break

                    if DISABLE and "visualize gradients of different losses":
                        import ipdb; ipdb.set_trace()
                        if False:
                            masks_from_pr.retain_grad()
                        masks_from_pr.grad = None
                        loss_for_grad = loss_var_l2_zbuf
                        # loss_for_grad = loss_l2_masks
                        # loss_for_grad = loss_var_l2_zbuf
                        loss_for_grad.backward(retain_graph=True)
                        # my_utils.img_save('masks_from_pr.png', masks_from_pr.grad[0].abs().sum(dim=0).detach().cpu().numpy())
                        ix_for_viz = 0
                        my_utils.img_save(tensor_to_numpy(masks_from_pr)[ix_for_viz],'masks_from_pr.png')
                        my_utils.img_save(tensor_to_numpy(masks_from_pr.grad)[ix_for_viz],'masks_from_pr_grad.png')
                        g1 = masks_from_pr.grad/masks_from_pr.grad.max()
                        g2 = g1.abs()
                        my_utils.img_save(tensor_to_numpy(g2)[ix_for_viz],'masks_from_pr_grad2.png')
                    # import ipdb; ipdb.set_trace()
                elif ENABLE_MESH == 1:
                    # assert False
                    shifted_mesh_model = mesh_model.offset_verts(vert_offsets)
                    
                    L2_VERTS = float(os.environ.get('L2_VERTS', 0))
                    L2_VERTS = 10
                    DISTILL_AFTER = 0

                    vertsparam_for_l2_vertsparam = shifted_mesh_model.verts_packed()
                    vertsparam_for_l2_vertsparam.retain_grad()
                    loss_l2_vertsparam = (vertsparam_for_l2_vertsparam - vertsparam_for_l2_vertsparam.mean(dim=0,keepdim=True)).norm(dim=1).sum()
                    trends['loss_l2_vertsparam'].append(loss_l2_vertsparam.item())                    
                    from pytorch3d.loss import (
                        chamfer_distance, 
                        mesh_edge_loss, 
                        mesh_laplacian_smoothing, 
                        mesh_normal_consistency,
                    )
                    NEXT_MULTIPLY = 0.1
                    # w_edge = 1.0 * 10 * NEXT_MULTIPLY
                    # Weight for mesh normal consistency
                    # w_normal = 0.01 * 10 * NEXT_MULTIPLY
                    # Weight for mesh laplacian smoothing
                    # w_laplacian = 0.1  * 10 * NEXT_MULTIPLY     
                    
                    w_edge =  1e4
                    # Weight for mesh normal consistency
                    w_normal = 1e4
                    # Weight for mesh laplacian smoothing
                    w_laplacian = 1e0
                    # import ipdb; ipdb.set_trace()
                    loss_edge = mesh_edge_loss(shifted_mesh_model)
                    loss_normal = mesh_normal_consistency(shifted_mesh_model)
                    loss_laplacian = mesh_laplacian_smoothing(shifted_mesh_model, method="uniform")
                    loss_aux = (
                            # self.var_scale_l2 * loss_var_l2 + \
                            # self.var_scale_l1 * loss_var_l1 + \
                            
                            self.var_scale_l2 * loss_var_l2_other + \
                            self.var_scale_l1 * loss_var_l1_other + \
                            self.bn_reg_scale * loss_r_feature + \
                            self.l2_scale * loss_l2_other +\
                            # L2_PR* self.l2_scale * loss_l2_pr +\
                            0*(loss_edge * w_edge +\
                            loss_normal * w_normal +\
                            loss_laplacian * w_laplacian  +\
                            float(iteration_loc >= DISTILL_AFTER) *L2_VERTS * loss_l2_vertsparam)
                    )
                    # import ipdb; ipdb.set_trace()
                    
                else:
                    assert False

                if self.adi_scale!=0.0:
                    loss_aux += self.adi_scale * loss_verifier_cig
                # trends['main_loss_pr'].append(main_loss_pr.item())
                trends['main_loss_other'].append(main_loss_other.item())

                    

                if len(trends['pr_acc']) and  max(trends['pr_acc']) == 1 and pr_acc < 0.1:
                    for lname in ['pr_acc','main_loss_pr','main_loss_other','mesh_volume','n_pts','dist_min','loss_var_l2','loss_l2_vertsparam','loss_l2_pr','loss_var_l2_masks','loss_var_l2_zbuf','loss_var_l2_first','tv_3d','loss_l2_masks','loss_l2_zbuf','loss_modelnet']:
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
                        inputs_from_pr1,target_pose1,surface_z1 = render(
                                                    pr_model,R1,T1,focal_length,device,
                                                    ENABLE_PR,ENABLE_MESH,
                                                    pr_args=pr_args
                                                    )
                        inputs_from_pr1 = inputs_from_pr1.float()
                        inputs_from_pr1 = inputs_from_pr1.permute(0,3,1,2)
                        inputs_from_pr1 = normalize(inputs_from_pr1,inplace=False)
                        outputs1 = net_teacher(inputs_from_pr1)
                        pr_acc1 = (outputs1.max(dim=1)[0] ==  outputs1[range(NVIEW),targets]).float().mean()
                        print(pr_acc1)          
                    run_for_arbitrary_pose()
                    import ipdb; ipdb.set_trace()
                # trends['pr_acc'].append(pr_acc.item())
                """
                next? actual classification oss?
                """     
                
                other_factor = 1e-5
                pr_factor = 1e-5
                if ENABLE_MESH == 1:
                    pr_factor = 1e0
                inputs_other_for_pr_loss = (other_factor)*inputs_other + max((1 - other_factor),0)*inputs_other.detach()
                inputs_from_pr_for_pr_loss = (pr_factor)*inputs_from_pr + max((1 - pr_factor),0)*inputs_from_pr.detach()
                # main_loss_pr = ((inputs_from_pr_for_pr_loss -  inputs_other_for_pr_loss)**2).sum()
                # import ipdb; ipdb.set_trace()
                main_loss_pr = (
                    (
                        masks_from_pr.detach() * (inputs_from_pr_for_pr_loss -  inputs_other_for_pr_loss)
                    )**2).sum()
                
                trends['main_loss_pr'].append(main_loss_pr.item())
                os.environ['INVERT_DGCNN'] = '1'
                print(colorful.yellow_on_red('hard coding INVERT_DGCNN'))
                if os.environ.get('INVERT_DGCNN',False) == '1':
                    # import ipdb;ipdb.set_trace()
                    loss = 1000*loss_modelnet+ 0*(
                    #1e-3
                    float(iteration_loc>=DISTILL_AFTER)*float(os.environ.get('PR_LOSS',1e0))* self.main_loss_multiplier * main_loss_pr + 
                    self.main_loss_multiplier * main_loss_other + 
                    loss_aux)
                    trends['loss_modelnet'].append(loss_modelnet.item())
                else:
                    loss = (
                    #1e-3
                    float(iteration_loc>=DISTILL_AFTER)*float(os.environ.get('PR_LOSS',1e0))* self.main_loss_multiplier * main_loss_pr + 
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
                if FIXTHIS:
                    modelnet_optimizer.step()
                
                # import ipdb; ipdb.set_trace()
                vertsparam_for_l2_vertsparam.grad
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
                    if USE_TRANSPARENCY:
                        alphas.data.copy_(alphas.data.clamp(0,1))
                if False:
                    for g in optimizer.param_groups:
                        g["lr"] = lr * decay_rate ** (iteration / decay_steps)
                # clip color outlayers

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()
                print(colorful.orange(f"{self.prefix}"))
                print(colorful.orange(f"{PDB_PORT}"))
                print(colorful.red('make losses with surface_z'))
                # import ipdb; ipdb.set_trace()                
                if (save_every > 0)  and any([iteration % save_every==0  , 
                                            #   iteration in [10,20,30,40,50,60,70,80,90]
                                              ]) or os.environ.get('SAVE_ALL',False) == '1' or USE_TRANSPARENCY:
                    if os.environ.get('SAVE_ALL',False) == '1':
                        import ipdb; ipdb.set_trace()
                    if local_rank==0:
                        # print('pickle save the images for running tests etc. later')
                        # import ipdb; ipdb.set_trace()
                        if False:
                            import pickle
                            with open('{}/best_images/output_{:05d}_gpu_{}.pkl'.format(self.prefix,
                                                                                            iteration // save_every,
                                                                                            local_rank),'wb') as f:
                                pickle.dump(tensor_to_numpy(inputs),f)
                        for lname in ['pr_acc','main_loss_pr','main_loss_other','mesh_volume','n_pts','dist_min','loss_var_l2','loss_l2_vertsparam','loss_l2_pr','loss_var_l2_masks','loss_var_l2_zbuf','loss_var_l2_first','tv_3d','loss_l2_masks','loss_l2_zbuf','loss_l2_surface_z','loss_var_l2_surface_z','loss_modelnet']:
                            my_utils.save_plot(trends[lname],lname,f'{os.path.join(self.prefix,"best_images",lname)}.png')


                        outputpath_pr = '{}/best_images/output_pr_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank)              
                        vutils.save_image(inputs_from_pr_pre_norm,
                                          outputpath_pr,
                                          normalize=False, scale_each=False, nrow=int(10))           
                        
                        maskpath = '{}/best_images/mask_pr_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank)
                        vutils.save_image(masks_from_pr,
                                          maskpath,
                                          normalize=False, scale_each=False, nrow=int(10))           


                        outputpath = '{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank)                            
                        
                        vutils.save_image(inputs_other,
                                          outputpath,
                                          normalize=True, scale_each=True, nrow=int(10))     
                        if surface_z is not None:
                            surface_z_path = '{}/best_images/surface_z_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                            iteration // save_every,
                                                                                            local_rank)
                            vutils.save_image(surface_z,
                                            surface_z_path,
                                            normalize=False, scale_each=False, nrow=int(10))           
                            
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","output_latest.png")}')
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","output_pr_latest.png")}')                        
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","mask_pr_latest.png")}')
                        os.system(f'unlink {os.path.join(self.prefix,"best_images","surface_z_latest.png")}')
                        # import ipdb; ipdb.set_trace()
                        os.system(f'ln -s {os.path.abspath(outputpath)} {os.path.join(self.prefix,"best_images","output_latest.png")}')
                        os.system(f'ln -s {os.path.abspath(outputpath_pr)} {os.path.join(self.prefix,"best_images","output_pr_latest.png")}')                       
                        os.system(f'ln -s {os.path.abspath(maskpath)} {os.path.join(self.prefix,"best_images","mask_pr_latest.png")}')                       
                        os.system(f'ln -s {os.path.abspath(surface_z_path)} {os.path.join(self.prefix,"best_images","surface_z_latest.png")}')                       
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
