from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
)           
from pytorch3d.renderer import cameras as p3dcameras
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
import os
import torch
import colorful
def render(
           pr_model,R,T,focal_length,device,
           ENABLE_PR,ENABLE_MESH,
           mesh_model=None,vert_offsets=None,pr_args=None,
           alphas = None,
           ):
    if ENABLE_PR:
        return render_points(pr_model,R,T,focal_length,device,alphas =alphas)
    elif ENABLE_MESH:
        return render_mesh(mesh_model,vert_offsets,
                pr_args,R,T,focal_length,device)
        
    
def render_points(pr_model,R,T,focal_length,device,alphas=None):
    
    NVIEW = R.shape[0]
    inputs_from_pr = []
    surface_z = None
    if True:
        cameras = PerspectiveCameras(
                    # focal_length=K[0][0] / K[0][2],
                    focal_length = focal_length,
                        device=device, 
                        R=R, T=T)
        Pmat = p3dcameras._get_sfm_calibration_matrix(cameras._N,cameras.device,cameras.focal_length,cameras.principal_point,orthographic=False)
        K = Pmat[0,[0,1,3],:3]
        target_pose = torch.zeros(R.shape[0],4,4,device=device)
        target_pose[:,:3, :3] = R
        target_pose[:,:3, -1] = T
        if os.environ.get('ONLYBASE',False) == '1':
            pr_model.onlybase = True
            # print(colorful.red("using onlybase"))
        if alphas is None:
            # assert False
            # import ipdb; ipdb.set_trace()
            inputs_from_pr = pr_model.forward_for_deepinversion(target_pose,K)
        else:             
            from torch import nn
            class MyPointsRenderer(nn.Module):
                """
                A class for rendering a batch of points. The class should
                be initialized with a rasterizer and compositor class which each have a forward
                function.
                """

                def __init__(self, rasterizer, compositor) -> None:
                    super().__init__()
                    self.rasterizer = rasterizer
                    self.compositor = compositor

                def to(self, device):
                    # Manually move to device rasterizer as the cameras
                    # within the class are not of type nn.Module
                    self.rasterizer = self.rasterizer.to(device)
                    self.compositor = self.compositor.to(device)
                    return self

                def forward(self, point_clouds, **kwargs) -> torch.Tensor:
                    fragments = self.rasterizer(point_clouds, **kwargs)

                    # Construct weights based on the distance of a point to the true point.
                    # However, this could be done differently: e.g. predicted as opposed
                    # to a function of the weights.
                    r = self.rasterizer.raster_settings.radius
                    custom_alphas = kwargs.pop('custom_alphas',None)
                    dists2 = fragments.dists.permute(0, 3, 1, 2)
                    
                    weights = 1 - dists2 / (r * r)
                    fragments_idx_long = fragments.idx.long()                    
                    if False:
                        # https://github.com/dmlc/dgl/issues/3729
                        alphas2 = custom_alphas[fragments_idx_long]
                    else:
                        # https://github.com/dmlc/dgl/issues/3729
                        # import ipdb; ipdb.set_trace()
                        custom_alphas = custom_alphas
                        fragments_idx_long = fragments_idx_long
                        alphas2 = torch.index_select(custom_alphas,0,fragments_idx_long.flatten().clamp(0)).reshape(fragments_idx_long.shape)
                    alphas2[fragments_idx_long == -1] = 0
                    alphas2 = alphas2.permute(0, 3, 1, 2)
                    alphas2 = alphas2.contiguous()
                    if custom_alphas is not None:
                        weights = alphas2 * weights
                    # else:
                    #     assert False
                    images = self.compositor(
                        fragments.idx.long().permute(0, 3, 1, 2),
                        weights,
                        point_clouds.features_packed().permute(1, 0),
                        **kwargs,
                    )
                    # new_zbuf = fragments.zbuf
                    images = self.compositor(
                        fragments.idx.long().permute(0, 3, 1, 2),
                        weights,
                        point_clouds.features_packed().permute(1, 0),
                        **kwargs,
                    )
                    images = images.permute(0, 2, 3, 1)
                    
                    transmission = 1 - alphas2
                    log_transmission = torch.log(transmission.clamp(1e-6))
                    cumsum_log_transmission = log_transmission.cumsum(dim=1)
                    transmission_ = torch.exp(cumsum_log_transmission)
                    transmission = torch.ones_like(transmission_)
                    transmission[:,1:] = transmission_[:,:-1]
                    visibility = transmission * alphas2
                    visibility = visibility.permute(0,2,3,1)
                    surface_z = (fragments.zbuf * visibility).sum(dim=-1).unsqueeze(-1)
                    # import ipdb; ipdb.set_trace()
                    # permute so image comes at the end
                    
                    # import ipdb; ipdb.set_trace()
                    return images,surface_z
            # print('A')
            if 'copied from coremodel':
                assert target_pose.ndim == 3
                R, T = (target_pose[:, :3, :3]), target_pose[:, :3, -1]
                """
                if DISABLED:
                    R, T = R, -(T[: ,None ,:] @ R)[: ,0]
                """
                focal = K[0,0]
                assert K[0,0] == K[1,1]

                R_for_camera,T_for_camera = R,T
                # print(dict(R=R_for_camera, T=T_for_camera))
                cameras = PerspectiveCameras(focal_length=focal,
                                            device=device, R=R_for_camera, T=T_for_camera)
                rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pr_model.raster_settings)
                renderer = MyPointsRenderer(
                    rasterizer=rasterizer,
                    compositor=AlphaCompositor(),
                    
                )

                """
                for depth https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/points/rasterizer.py
                """
                vertsparam = pr_model.vertsparam
                sh_param = pr_model.sh_param
                if pr_model.symmetry == 'z':
                    n_pts_half = vertsparam.shape[0]
                    vertsparam = vertsparam.repeat(2,1)
                    vertsparam[n_pts_half:,-1] = - vertsparam[n_pts_half:,-1]
                    sh_param = sh_param.repeat(2,1)
                point_cloud = Pointclouds(points=vertsparam[None].repeat(target_pose.shape[0],1,1), features=pr_model.sh_param[None].repeat(target_pose.shape[0],1,1))
                if True:
                    
                    feat,surface_z = renderer(point_cloud,custom_alphas=alphas.repeat(NVIEW),eps=1e-8)
                    feat = feat.flip(1)
                    surface_z = surface_z.flip(1)
                    # print('B')
                    feat = feat[...,:3]
                inputs_from_pr = feat
            # import ipdb; ipdb.set_trace()
        target_pose = target_pose[0]
        # return inputs_from_pr,target_pose
        
    # inputs_from_pr.append(inputs_from_pri)
    # import ipdb; ipdb.set_trace()            
    if False:
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
    return inputs_from_pr,target_pose,surface_z
                    

def render_mesh(mesh_model,vert_offsets,
                pr_args,R,T,focal_length,device):
    inputs_from_pr = []
    NVIEW = R.shape[0]
    
    if True:
        cameras = PerspectiveCameras(
                # focal_length=K[0][0] / K[0][2],
                focal_length = focal_length,
                    device=device, 
                    R=R, T=T)        
        from pytorch3d.renderer import (
            # PerspectiveCameras,
            # PointsRasterizationSettings,
            # PointsRenderer,
            # PointsRasterizer,
            RasterizationSettings,
            MeshRasterizer,
            # MeshRasterizationSettings,
            # AlphaCompositor,
            MeshRenderer, 
            HardPhongShader,
            # TexturesVertex,
            BlendParams
        )             
        import pytorch3d.structures                       
        import pytorch3d.renderer
        raster_settings = RasterizationSettings(
                # bin_size=23,
                image_size=pr_args.img_s,
                # radius=splatting_r,
                # points_per_pixel=raster_n,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        
        # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        blend_params = BlendParams(background_color = [0,0,0])
        mesh_renderer = MeshRenderer(
            rasterizer=rasterizer,  # We'll use the default rasterizer
            shader=HardPhongShader(cameras=cameras,blend_params=blend_params,device=device),  # We'll use the HardPhongShader to shade the mesh
            # device=device   
        )
        T = mesh_model.textures.verts_features_padded()
        T = T.repeat(NVIEW,1,1)
        T = pytorch3d.renderer.mesh.TexturesVertex(T)

        F = mesh_model.faces_list()[0]
        V = mesh_model.verts_list()[0]
        F = F[None].repeat(NVIEW,1,1)
        V = V[None].repeat(NVIEW,1,1)
        mesh_model2 = pytorch3d.structures.Meshes(verts=V, faces=F, textures=T)

        shifted_mesh_model = mesh_model2.offset_verts(vert_offsets.repeat(NVIEW,1))    
        # import ipdb; ipdb.set_trace()
        inputs_from_pr  = mesh_renderer(shifted_mesh_model)
        # inputs_from_pr.append(inputs_from_pri)
        inputs_from_pr = inputs_from_pr[...,:3]
    if False:
        for ni in range(NVIEW):
            Ri,Ti = R[ni],T[ni]
            cameras = PerspectiveCameras(
                    # focal_length=K[0][0] / K[0][2],
                    focal_length = focal_length,
                        device=device, 
                        R=Ri[None], T=Ti[None])
            """
            Pmat = p3dcameras._get_sfm_calibration_matrix(cameras._N,cameras.device,cameras.focal_length,cameras.principal_point,orthographic=False)
            K = Pmat[0,[0,1,3],:3]
            target_pose = torch.zeros(4,4,device=device)
            target_pose[:3, :3] = Ri
            target_pose[:3, -1] = Ti
            """
            from pytorch3d.renderer import (
                # PerspectiveCameras,
                # PointsRasterizationSettings,
                # PointsRenderer,
                # PointsRasterizer,
                RasterizationSettings,
                MeshRasterizer,
                # MeshRasterizationSettings,
                # AlphaCompositor,
                MeshRenderer, 
                HardPhongShader,
                # TexturesVertex
            )                            
            raster_settings = RasterizationSettings(
                # bin_size=23,
                image_size=pr_args.img_s,
                # radius=splatting_r,
                # points_per_pixel=raster_n,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            mesh_renderer = MeshRenderer(
                rasterizer=rasterizer,  # We'll use the default rasterizer
                shader=HardPhongShader(cameras=cameras,device=device),  # We'll use the HardPhongShader to shade the mesh
                # device=device   
            )
            shifted_mesh_model = mesh_model.offset_verts(vert_offsets)
            if False:
                # shifted_mesh_model = mesh_model.offset_verts(torch.rand_like(vert_offsets)*0.5)
                from delaunay import render_mesh
                _=render_mesh(shifted_mesh_model,focal_length)
            inputs_from_pri  = mesh_renderer(shifted_mesh_model)
            inputs_from_pr.append(inputs_from_pri)
        inputs_from_pr = torch.cat(inputs_from_pr,dim=0)
        inputs_from_pr = inputs_from_pr[...,:3]
    return inputs_from_pr,None

def get_RT(elev,azim,dist):
    # import ipdb; ipdb.set_trace()
    R, T = look_at_view_transform(dist=dist,azim=azim,elev=elev,at=((0,0,0,),))
    # import ipdb; ipdb.set_trace()
