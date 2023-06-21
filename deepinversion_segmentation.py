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
import os
from sparsity import setup_network_for_comprehensive_sparsity
import torch.nn.functional as F
tensor_to_numpy = lambda x: x.detach().cpu().numpy()
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
                 use_fp16=True, net_teacher=None, path="./gen_images/",
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
        torch.manual_seed(torch.cuda.current_device())

        self.net_teacher = net_teacher

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
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

        local_rank = torch.cuda.current_device()
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
        self.activations,self.noisy_activations,self.avg_angles,self.avg_mags = setup_network_for_comprehensive_sparsity(net_teacher,layer_type=nn.Conv2d,noise_mag=0)
    def get_images(self, net_student=None, targets=None):

        print("get_images call")

        net_teacher = self.net_teacher
        use_fp16 = self.use_fp16
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion

        # setup target labels
        if targets is None:
            #only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor([random.randint(0, 999) for _ in range(self.bs)]).to('cuda')
            if not self.random_label:
                if os.environ.get('SINGLE_CLASS',False):
                    # dog class
                    targets = [int(os.environ['SINGLE_CLASS'])]
                else:
                    # preselected classes, good for ResNet50v1.5
                    targets = [1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 277, 283, 292, 294, 309,
                            311,
                            325, 340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
                            967, 574, 487]
                    targets = targets[:self.bs]


                targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to('cuda')

        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        inputs = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device='cuda',
                             dtype=data_type)
        inputs_seg = torch.rand((self.bs, 1, img_original, img_original), requires_grad=True, device='cuda',
                             dtype=data_type)
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
                # iterations_per_layer = 200
                
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000
                
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
            
            # print('check what the setting_id is and if what happens if you remove color clipping?')
            # import ipdb; ipdb.set_trace()
            assert self.setting_id == 0,'add optimizer_seg to other settings'
            if self.setting_id == 0:
                #multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                optimizer_seg = optim.Adam([inputs_seg], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                if os.environ.get('NO_CLIP', False) == '1':
                    print(colorful.red("setting clip to False"))
                    do_clip = False
                else:
                    do_clip = True
            elif self.setting_id == 1:
                #2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
            elif self.setting_id == 2:
                #20k normal resolution the closes to the paper experiments for ResNet50
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)
                do_clip = False

            if use_fp16:
                static_loss_scale = 256
                static_loss_scale = "dynamic"
                _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                    inputs_seg_jit = pooling_function(inputs_seg)
                else:
                    inputs_jit = inputs
                    inputs_seg_jit = inputs_seg

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
                inputs_seg_jit = torch.roll(inputs_seg_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))
                    inputs_seg_jit = torch.flip(inputs_seg_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                optimizer_seg.zero_grad()
                net_teacher.zero_grad()                
                #===============================================================
                outputs = net_teacher(inputs_jit)
                
                outputs = self.network_output_function(outputs)
                
                # # R_cross classification loss
                # # print('check if targets are just indicators')
                # # import ipdb; ipdb.set_trace()
                loss = criterion(outputs, targets)

                if False:
                    mask_pyramid= []
                    for ia in range(len(self.activations)):
                        if ia in [0,11,24,43,52]:
                            mask_pyramid.append(F.interpolate(inputs_seg_jit, size=self.activations[ia][0].shape[-2:], mode='bilinear',antialias=True))
                else:
                    from image_pyramid import gaussian_pyramid
                    mask_pyramid = gaussian_pyramid(inputs_seg_jit, 
                                                    None, 
                                                    # downsampling_factor=2, 
                                                    sizes  = [a[0].shape[-2:] for ia,a in enumerate(self.activations) if ia in [0,11,24,43,52]],
                                                    antialias=True)
                    mask_pyramid = mask_pyramid[1:]
                    
                original_activations = [a[0].detach().clone() for ia,a in enumerate(self.activations) if ia in [0,11,24,43,52]]
                
                #===============================================================
                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                loss_seg_var_l1, loss_seg_var_l2 = get_image_prior_losses(inputs_seg_jit)
                loss_sharpness = torch.sum(
                    (inputs_seg_jit > 0.5).float() * (-inputs_seg_jit) + (inputs_seg_jit < 0.5).float() * (inputs_seg_jit)
                )
                
                areas = inputs_seg_jit.sum(dim=(1,2,3))
                areas_std = areas.std()
                # import ipdb; ipdb.set_trace()
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
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()
                #===============================================================

                
                # import ipdb; ipdb.set_trace()
                # make a mask pyramid by resizing the mask to the same size as the activations

                #calculate the l1 distortions between the activations and the masked activations
                
                # import ipdb; ipdb.set_trace()
                if 'masked loss' and True:
                    inputs_seg_jit2 = inputs_seg_jit
                    off1_seg = random.randint(-lim_0, lim_0)
                    off2_seg = random.randint(-lim_1, lim_1)
                    inputs_seg_jit2 = torch.roll(inputs_seg_jit, shifts=(off1_seg, off2_seg), dims=(2, 3))
                    
                    
                    from elp_masking import get_masked_input,PRESERVE_VARIANT,DELETE_VARIANT
                    # masked_inputs_jit_preserve,_ = get_masked_input(inputs_jit.detach(), inputs_seg_jit,variant=PRESERVE_VARIANT,num_levels=2)
                    masked_inputs_jit_preserve = torch.zeros_like(inputs_jit)
                    if True:
                        for ii in range(inputs_jit.shape[0]):
                            # import ipdb; ipdb.set_trace()
                            masked_inputs_jit_preserve[ii:ii+1],_ = get_masked_input(
                                inputs_jit[ii:ii+1].detach(), 
                                inputs_seg_jit2[ii:ii+1],
                                variant=PRESERVE_VARIANT,
                                num_levels=8)
                    else:
                        masked_inputs_jit_preserve = inputs_seg_jit*inputs_jit.detach()
                        
                    
                    masked_inputs_jit_delete,_ = get_masked_input(inputs_jit.detach(), inputs_seg_jit2,variant=DELETE_VARIANT,num_levels=8)
                    outputs_preserve = net_teacher(masked_inputs_jit_preserve)
                    outputs_preserve = self.network_output_function(outputs_preserve)
                    preserve_criterion = nn.CrossEntropyLoss(reduction='none')
                    # import ipdb; ipdb.set_trace()
                    assert isinstance(preserve_criterion,criterion.__class__)
                    loss_preserve = preserve_criterion(outputs_preserve, targets) * (-1*(outputs_preserve.argmax(dim=1) == targets).float() + 0.5)*2
                    
                    preserve_activations = [a[0] for ia,a in enumerate(self.activations) if ia in [0,11,24,43,52]]   
                    activation_distortions = []
                    for oa,pa,ma in zip(original_activations,preserve_activations,mask_pyramid):
                        d = torch.abs(oa - pa) * (((oa > 0) & (pa <0)) | ((oa < 0) & (pa >0))).float() * ma
                        d = d.sum()
                        activation_distortions.append(d)        
                    # loss_activation_distortion = sum([1e-5 * d for d in activation_distortions])
                    # import ipdb; ipdb.set_trace()
                    # [0,11,24,43,52]
                    loss_activation_distortion = 0e-4 * activation_distortions[0] + 0e-4 * activation_distortions[1] + 0e-3 * activation_distortions[3] + 1e-3 * activation_distortions[4]
                    loss_preserve = loss_preserve.mean()
                    # import ipdb; ipdb.set_trace()
                    loss_delete = 0
                    
                    if False:
                        outputs_delete = net_teacher(masked_inputs_jit_delete)
                        outputs_delete = self.network_output_function(outputs_delete)
                        loss_delete = criterion(outputs_delete, targets)
                else:
                    masked_inputs_jit_preserve = torch.zeros_like(inputs_jit)
                    loss_delete = 0
                    loss_preserve = 0
                #===============================================================
                
                # combining losses
                loss_aux = self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                           self.bn_reg_scale * loss_r_feature + \
                           self.l2_scale * loss_l2
                loss_seg_aux = loss_seg_var_l1 + 0*loss_seg_var_l2 - 0e-2*areas_std + 0e-4*loss_sharpness + loss_activation_distortion
                #===============================================================


                if self.adi_scale!=0.0:
                    loss_aux += self.adi_scale * loss_verifier_cig
                loss_seg = 1*(loss_preserve + loss_delete) + 1*loss_seg_aux
                loss = self.main_loss_multiplier * (loss ) + loss_aux  + 1*loss_seg

                if local_rank==0:
                    if iteration % save_every==0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print("loss_preserve", loss_preserve.item())
                        print("mask area", inputs_seg.mean().item())
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

                optimizer.step()
                if True:
                    optimizer_seg.step()

                # clip color outlayers
                inputs_seg.data = torch.clip(inputs_seg.data,0,1 )
                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=use_fp16)

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

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
                        vutils.save_image(inputs_seg,
                                          '{}/best_images/output_seg_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank),
                                          normalize=True, scale_each=True, nrow=int(10))
                        vutils.save_image(masked_inputs_jit_preserve,
                                          '{}/best_images/output_masked_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank),
                                          normalize=True, scale_each=True, nrow=int(10))

        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
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
        # for ADI detach student and add put to eval mode
        net_teacher = self.net_teacher

        use_fp16 = self.use_fp16

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()

        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).cuda()
            if use_fp16:
                targets = targets.half()

        self.get_images(net_student=net_student, targets=targets)

        net_teacher.eval()

        self.num_generations += 1
