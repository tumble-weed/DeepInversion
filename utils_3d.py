import torch
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    look_at_view_transform,
    TexturesVertex,
)              
from pytorch3d.structures import ( Pointclouds,
Meshes
                                  )
import numpy as np
import colorful
from my_renderer import render

def get_mask_and_zbuf_from_point_cloud(
    focal_length,
    pr_model,
    NVIEW,
    R,
    T,
    device
    ):
    # ENABLE_MESH = False
    # ENABLE_PR = True
    focal =focal_length
    p1 = (pr_model.vertsparam for _ in range(NVIEW))
    p1 = list(p1)
    p1 = torch.stack(p1,dim=0)
    dummy_texture = torch.ones_like(pr_model.sh_param[...,:3]).detach()
    # feat1 = (dummy_texture for _ in range(NVIEW))
    # feat1 = list(feat1)
    feat1 = dummy_texture[None].repeat(NVIEW,1,1)
    point_cloud_for_depth = Pointclouds(points=p1, features=feat1)
    if True:
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
            
            for_depth_rasterized = rasterizer(point_cloud_for_depth)
            
            zbuf = for_depth_rasterized.zbuf[...,0].flip(1).unsqueeze(1) 
            zmax = zbuf.view(*zbuf.shape[:2],-1).max(dim=-1)[0].detach()
            for jj in range(zbuf.shape[0]):
                zbuf[jj][zbuf[jj]==-1] = zmax[jj]
            # zmin = zbuf.min().detach()
            zmin = zbuf.view(*zbuf.shape[:2],-1).min(dim=-1)[0].detach()
            assert zmax.shape == (zbuf.shape[0],1)
            zbuf = (zmax[...,None,None] - zbuf)#/(zmax - zmin)
            # import ipdb; ipdb.set_trace()
            # assert False
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
    if True:
        print(colorful.khaki('using random views for zbuf'))
        elev_rand,azim_rand,dist_rand =  -45 + 90*np.random.random(NVIEW),0 + 360*np.random.random(NVIEW),1.2*np.ones(NVIEW)          
        Rrand, Trand = look_at_view_transform(dist=dist_rand,azim=azim_rand,elev=elev_rand,at=((0,0,0,),))
        cameras = PerspectiveCameras(focal_length=focal,
                    device=device, R=Rrand, T=Trand)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pr_model.raster_settings)
        
        for_depth_rasterized = rasterizer(point_cloud_for_depth)
        
        zbuf_rand = for_depth_rasterized.zbuf[...,0].flip(1).unsqueeze(1) 
        zbuf_randmax = zbuf_rand.view(*zbuf_rand.shape[:2],-1).max(dim=-1)[0].detach()
        for jj in range(zbuf_rand.shape[0]):
            zbuf_rand[jj][zbuf_rand[jj]==-1] = zbuf_randmax[jj]
        # zmin = zbuf.min().detach()
        zbuf_randmin = zbuf_rand.view(*zbuf_rand.shape[:2],-1).min(dim=-1)[0].detach()
        assert zbuf_randmax.shape == (zbuf_rand.shape[0],1)
        zbuf_rand = (zbuf_randmax[...,None,None] - zbuf_rand)#/(zmax - zmin)
        # import ipdb; ipdb.set_trace()
        # assert False                                
        
        

def get_mask_and_zbuf_from_mesh(
    focal_length,
    mesh_model,

    R,
    T,
    device
    ):
    ENABLE_MESH = True
    ENABLE_PR = False
    # assert False,'pr_args not implemented'
    inputs_from_pr,target_pose,surface_z = render(
        None,R,T,focal_length,device,
        ENABLE_PR,ENABLE_MESH,
        mesh_model=mesh_model,vert_offsets=vert_offsets,pr_args=pr_args,
        )
    nvert = vert_offsets.shape[0]
    mask_texture = torch.ones(nvert,3,device=device,requires_grad=True)
    
    mask_textures_obj = TexturesVertex(verts_features=mask_texture[None])
    # mesh_model.textures = textures_obj

    mask_mesh_model  = Meshes(verts=[mesh_model.verts_packed()], faces=[mesh_model.faces_packed()],textures=mask_textures_obj)                        
    
    masks_from_pr,_,surface_z =render(
        None,R,T,focal_length,device,
        ENABLE_PR,ENABLE_MESH,
        mesh_model=mask_mesh_model,vert_offsets=vert_offsets,pr_args=pr_args,
        )
    masks_from_pr = masks_from_pr.permute(0,3,1,2)
    assert masks_from_pr.max() <= 1.
    assert masks_from_pr.min() >= 0.
    # import ipdb; ipdb.set_trace()    
    
def visualize_pr_model(pr_model,pr_args):
    from view_sampling import sample_view_params
    focal_length = 0.5
    iteration_loc = None
    iteration = None
    trends = {}
    device = pr_model.vertsparam.device
    elev,azim,dist = sample_view_params(
    pr_model,
    iteration_loc,
    iteration,
    trends,
    device,
    False,
    None,
    None,
    UNIFORM_AZIM = 'UNIFORM_AZIM' in os.environ
    )
    #==============================================================================================
    from my_renderer import render
    from pytorch3d.renderer import (
        look_at_view_transform,
        PerspectiveCameras,
    )                      
    # import ipdb; ipdb.set_trace()
    R, T = look_at_view_transform(dist=dist,azim=azim,elev=elev,at=((0,0,0,),))
    # import ipdb; ipdb.set_trace()
    #==============================================================================================
    
    inputs_from_pr,target_pose,surface_z = render(
        pr_model,R,T,focal_length,device,
        True,False,
        pr_args=pr_args, alphas = None
        )
    inputs_from_pr = inputs_from_pr.permute(0,3,1,2)
    inputs_from_pr0 = inputs_from_pr
    return inputs_from_pr0