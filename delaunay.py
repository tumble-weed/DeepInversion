import torch
import scipy
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/delaunay.html
r"""Computes the delaunay triangulation of a set of points
(functional name: :obj:`delaunay`)."""
def calculate_delaunay( pos):
    device = pos.device
    if pos.size(0) < 2:
        edge_index = torch.tensor([], dtype=torch.long,
                                        device=device).view(2, 0)
    if pos.size(0) == 2:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long,
                                        device=device)
    elif pos.size(0) == 3:
        face = torch.tensor([[0], [1], [2]], dtype=torch.long,
                                    device=device)
    if pos.size(0) > 3:
        pos = pos.cpu().numpy()
        # tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
        tri = scipy.spatial.ConvexHull(pos)
        
        print(tri.simplices.shape)
        face = torch.from_numpy(tri.simplices)

        face = face.t().contiguous().to(device, torch.long)

        
    return face
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
def mesh_smoothness_loss(vertsparam):    
    faces = calculate_delaunay( vertsparam)
    mesh = Meshes(verts=[vertsparam], faces=[faces.T])
    # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    loss = mesh_laplacian_smoothing(mesh)
    return loss
from pytorch3d.renderer import (
    PerspectiveCameras,
    # PointsRasterizationSettings,
    # PointsRenderer,
    # PointsRasterizer,
    RasterizationSettings,
    MeshRasterizer,
    # MeshRasterizationSettings,
    AlphaCompositor,
    MeshRenderer, 
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
)                      
from pytorch3d.renderer import cameras as p3dcameras
import my_utils
def get_mesh(vertsparam,focal=None):
    device = vertsparam.device
    # vertsparam = pr_model.vertsparam


    faces = calculate_delaunay( vertsparam.detach())
    # verts_rgb = torch.zeros_like(vertsparam[...,:3])[None]  # (1, V, 3)
    # verts_rgb[...,0] = 1.0
    # textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(verts=[vertsparam], faces=[faces.T])
    return mesh

def render_mesh(mesh,focal):
    device = mesh.device
    img_s = 256
    splatting_r = 0.015
    raster_n = 15
    raster_settings = RasterizationSettings(
                # bin_size=23,
                image_size=img_s,
                # radius=splatting_r,
                # points_per_pixel=raster_n,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

    R, T = look_at_view_transform(eye=[(0,0,2)],at=((0,0,0,),))
    cameras = PerspectiveCameras(focal_length=focal,
                device=device, R=R, T=T)
    Pmat = p3dcameras._get_sfm_calibration_matrix(cameras._N,cameras.device,cameras.focal_length,cameras.principal_point,orthographic=False)
    K = Pmat[0,[0,1,3],:3]            
    # focal = K[0,0]
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    renderer = MeshRenderer(
        rasterizer=rasterizer,  # We'll use the default rasterizer
        shader=HardPhongShader(cameras=cameras,device=device),  # We'll use the HardPhongShader to shade the mesh
        # device=device

        
    )

    images = renderer(mesh)
    my_utils.img_save2(images,'mesh_rendered.png',syncable=False)
    return images

def render_as_mesh(vertsparam,focal=None):
    device = vertsparam.device
    # vertsparam = pr_model.vertsparam

    verts_rgb = torch.zeros_like(vertsparam[...,:3])[None]  # (1, V, 3)
    verts_rgb[...,0] = 1.0
    faces = calculate_delaunay( vertsparam)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(verts=[vertsparam], faces=[faces.T],textures=textures)
    images = render_mesh(mesh,focal)
    return images,mesh
def get_mesh_volume(mesh):
    """
    https://stackoverflow.com/a/12423555
    """
    faces_packed = mesh.faces_packed()
    verts_packed = mesh.verts_packed()
    v0 = verts_packed[faces_packed[ :, 0], :]
    v1 = verts_packed[faces_packed[ :, 1], :]
    v2 = verts_packed[faces_packed[ :, 2], :]
    cross_12 = torch.cross(v1,v2,dim=-1)
    volumes = torch.einsum('ij,ik->i',v0, cross_12)/6
    # according to below, you need to take the absolute value after the sum
    # https://stackoverflow.com/a/1568551
    
    volume = volumes.sum().abs()
    return volume
    
"""
# test volume
# Create a PyTorch3D Meshes object.
import torch
import pytorch3d
import pytorch3d.structures
verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
faces = torch.tensor([[0, 1, 2]])
meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])

# Get the packed vertices and faces of the mesh.
verts_packed = meshes.verts_packed()
faces_packed = meshes.faces_packed()

# Compute the face normals of the mesh.
# face_normals = pytorch3d.ops.mesh_ops.mesh_normal(verts_packed, faces_packed)
face_normals = meshes.faces_normals_packed()

"""