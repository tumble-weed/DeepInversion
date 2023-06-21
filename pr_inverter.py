from argparse import Namespace
import torch
import colorful
from torch import optim
from my_renderer import render

from pytorch3d.renderer import (
            look_at_view_transform,
            PerspectiveCameras,
        )                      
from view_sampling import sample_view_params
from pytorch3d.renderer import cameras as p3dcameras
from my_renderer import render

from refine_point_cloud import repeat_pts, remove_out
import os
# DONE = False
if False:
    def get_pr_args():
        #=======================================================================================
        do_clip = True
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
        return pr_args


    def sample_view(
            focal_length,
            ENABLE_MESH,
            ENABLE_PR,
            N_AL_TRAIN,
            device):

        view_info = {}
        #==============================================================================================
        elev,azim,dist = sample_view_params(
            pr_model,
            iteration_loc,
            iteration,
            trends,
            self.device,
            self.ENABLE_MESH,
            self.N_AL_TRAIN,
            view_errors,
            UNIFORM_AZIM = 'UNIFORM_AZIM' in os.environ
            )
        R, T = look_at_view_transform(dist=dist,azim=azim,elev=elev,at=((0,0,0,),))
        view_info = dict(
            elev = elev,
            azim = azim,
            dist = dist,
            R = R,
            T = T,
            focal_length = focal_length
        )
        return view_info

class GetView():
    def forward(self,pr_model,
        trends,view_errors,
        iteration,iteration_loc,extra_info = {}):

        #==============================================================================================
        inputs_from_pr,target_pose,surface_z = render(
            pr_model,R,T,focal_length,device,
            ENABLE_PR,ENABLE_MESH,
            pr_args=pr_args, alphas = alphas
            )
        surface_z = surface_z.permute(0,3,1,2)
#==============================================================================================
from pytorch3d.utils import ico_sphere
from pytorch3d.structures  import Meshes
from pytorch3d.renderer import (
    TexturesVertex
)            
class MeshInverter():
    def __init__(self,device):
        # pr_model =  None

        self.mesh_model = ico_sphere(5, device)
        # import ipdb; ipdb.set_trace()
        self.vert_offsets = torch.full(self.mesh_model.verts_packed().shape, 0.0, device=device, requires_grad=True)
        nvert = self.vert_offsets.shape[0]
        texture = torch.rand(nvert,3,device=device,requires_grad=True)
        textures_obj = TexturesVertex(verts_features=texture[None])
        # mesh_model.textures = textures_obj
        
        self.mesh_model  = Meshes(verts=[self.mesh_model.verts_packed()], faces=[self.mesh_model.faces_packed()],textures=textures_obj)
        # pr_optimizer = torch.optim.SGD([vert_offsets,texture], lr=1e-5, momentum=0.9)
        
        self.optimizer = torch.optim.Adam([
    {'params': self.vert_offsets, 'lr': 1e-3*1e-1*3*1e-1},
    {'params': texture, 'lr':  0.1*1e-1*1e1}])
#==============================================================================================

class PRInverter():
    def zero_grad(self):
        self.optimizer.zero_grad()
        pass
    def get_auxiliary_loss(self,trends):
        loss_var_l1, loss_var_l2 = get_image_prior_losses(gaussian_blur(inputs_from_pr, 31, sigma=None))        
        trends['loss_var_l2'].append(loss_var_l2.item())
        
        _,loss_var_l2_masks = get_image_prior_losses(gaussian_blur(masks_from_pr, 11, sigma=None))
        trends['loss_var_l2_masks'].append(loss_var_l2_masks.item())       

        print(colorful.salmon("using random zbuf in prior losses"))
        _,loss_var_l2_zbuf = get_image_prior_losses(gaussian_blur(zbuf_rand, 11, sigma=None))
        
        trends['loss_var_l2_zbuf'].append(loss_var_l2_zbuf.item())
        
        
        loss_l2_zbuf = ((zbuf_rand)**2).sum()
        trends['loss_l2_zbuf'].append(loss_l2_zbuf.item())                             
    def refine(self,):
        REFINE_AFTER = os.environ.get('REFINE_AFTER',None)
        if REFINE_AFTER == '':
            REFINE_AFTER = None
        REFINE_AFTER =  int(REFINE_AFTER) if REFINE_AFTER is not None else REFINE_AFTER
        REFINE_EVERY = os.environ.get('REFINE_EVERY',None)
        if REFINE_EVERY == '':
            REFINE_EVERY = None                
        REFINE_EVERY = int(REFINE_EVERY) if REFINE_EVERY is not None else REFINE_EVERY
        if REFINE_AFTER is not None:
            assert False, 'untested'
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
        
#===========================================================
        
    def __init__(self,lr,trends,device):
        self.lr = lr
        self.pr_args = get_pr_args()
        self.pr_model = CoreModel(pr_args,STANDARD=False,init_mode='random').to(device)
        self.pr_model.onlybase = True
        self.trends = trends
        self.set_optimizer(
        self.pr_model.sh_param, 
        self.pr_model.vertsparam, 
        lr1=(10**(-1*float(targets[0]==933) ))*self.pr_args.lr1, lr2=self.pr_args.lr2,lrexp=self.pr_args.lrexp,lr_s=self.pr_args.lr_s)
        self.do_clip = True
        self.pr_model.train()
        pass
    def set_optimizer(
                    self,
                    # pr_model, 
                    sh_params,
                    other_params,
                    lr1=3e-3, lr2=8e-4,lrexp=0.93,lr_s=0.03):
        # sh_list = [name for name, params in pr_model.named_parameters() if 'sh' in name]
        # sh_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in sh_list,
        #                         pr_model.named_parameters()))))
        # other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in sh_list,
        #                         pr_model.named_parameters()))))
        self.optimizer = torch.optim.Adam([
            {'params': sh_params, 'lr': lr1},
            {'params': other_params, 'lr': lr2}])
        lr_scheduler = torch.optim.self.lr_scheduler.ExponentialLR(self.optimizer, lrexp, -1)
        # self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
def render(pr_model,view_info,ENABLE_PR,ENABLE_MESH,pr_args,alphas=None):
    # tracer.run('main()')
    inputs_from_pr,target_pose,surface_z = render(
        pr_model,
        view_info['R'],view_info['T'],view_info['focal_length'],
        device,
        ENABLE_PR,ENABLE_MESH,
        pr_args=pr_args, alphas = alphas
        )
    surface_z = surface_z.permute(0,3,1,2)
    # inputs_from_pr,target_pose =  tracer.run('''render(
    #     pr_model,R,T,focal_length,device,
    #     ENABLE_PR,ENABLE_MESH,
    #     pr_args=pr_args, alphas = alphas
    #     )''')
    
    render_info = dict(
        inputs_from_pr = inputs_from_pr,
        target_pose = target_pose,
        surface_z = surface_z,
    )

class TransparencyInverter(PRInverter):
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        pass    
    def __init__(self,lr,trends,device):
        super(TransparencyInverter,self).__init__(lr,trends,device)
        self.alphas = torch.ones(self.pr_model.vertsparam[:,0].shape).float().to(device).requires_grad_(True)
        # self.set_optimizer(
        # self.pr_model.sh_param, 
        # self.pr_model.vertsparam, 
        # lr1=(10**(-1*float(targets[0]==933) ))*self.pr_args.lr1, lr2=self.pr_args.lr2,lrexp=self.pr_args.lrexp,lr_s=self.pr_args.lr_s)        
        
        """
        pr_optimizer, pr_lr_scheduler = set_optimizer(
                                            # pr_model, 
                                            pr_model.sh_param, 
                                            other_param, 
                                                                  
lr1=(10**(-1*float(targets[0]==933) ))*pr_args.lr1, lr2=pr_args.lr2,lrexp=pr_args.lrexp,lr_s=pr_args.lr_s)
        """

        self.alpha_optimizer = optim.Adam([self.alphas], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
        self.do_clip = True     
        
    def forward(self,):
        pass
    def get_auxiliary_loss(self,zero_grad=True):
        if zero_grad:
            self.optimizer.zero_grad()
        if False:
            _,loss_var_l2_zbuf = get_image_prior_losses(gaussian_blur(zbuf, 11, sigma=None))
            
            trends['loss_var_l2_zbuf'].append(loss_var_l2_zbuf.item())
            
            
            loss_l2_zbuf = ((zbuf)**2).sum()
            trends['loss_l2_zbuf'].append(loss_l2_zbuf.item())
        else:
            print(colorful.salmon("using random zbuf in prior losses"))
            _,loss_var_l2_zbuf = get_image_prior_losses(gaussian_blur(zbuf_rand, 11, sigma=None))
            
            trends['loss_var_l2_zbuf'].append(loss_var_l2_zbuf.item())
            
            
            loss_l2_zbuf = ((zbuf_rand)**2).sum()
            trends['loss_l2_zbuf'].append(loss_l2_zbuf.item())                    

        loss_l2_other = 0
        if inputs_other.shape[0]>0:
            loss_l2_other = torch.norm(inputs_for_prior_jit.view(inputs_jit.shape[0], -1), dim=1).mean()
        if True:
            loss_l2_pr = torch.norm(inputs_from_pr.reshape(inputs_from_pr.shape[0], -1), dim=1).mean()
            trends['loss_l2_pr'].append(loss_l2_pr.item())
            loss_l2_masks = torch.norm(masks_from_pr.reshape(masks_from_pr.shape[0], -1), dim=1).mean()
            # loss_l2_masks = torch.abs(masks_from_pr.reshape(masks_from_pr.shape[0], -1)).sum(dim=1).mean()
            trends['loss_l2_masks'].append(loss_l2_masks.item())
        else:
            """
            # inputs_jit_pr = inputs_jit[:NVIEW]
            loss_l2_pr = torch.norm((inputs_from_pr*(inputs_from_pr.abs()>1).float() ).view(inputs_from_pr.shape[0], -1), dim=1).mean()
            """
        #=========================
        
    def forward_and_loss(self):
        self.forward
        self.get_auxiliary_loss 
        pass

    def step(self,):
        self.optimizer.step()
        pass
    pass