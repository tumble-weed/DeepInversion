import torch
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    # mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
def mesh_edge_loss(meshes, target_length: float = 0.0,order=2):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_edge_loss.html
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    if order == 2:
        if False:
            loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
        else:
            obs_lens = (v0 - v1).norm(dim=1, p=2)
            log_obs_lens = torch.log(obs_lens)
            mean, std = torch.mean(log_obs_lens), torch.std(log_obs_lens)
            standardized = (log_obs_lens - mean) / std
            penalize_mask = (standardized.abs() > 2)
            loss = standardized[penalize_mask]**2
            weights = weights[penalize_mask]
    elif order == 1:
        loss = ((v0 - v1).norm(dim=1, p=2) - target_length).abs()
    loss = loss * weights

    return loss.sum() / N
class MeshModel(torch.nn.Module):
    def __init__(self,levels=2,n_mesh=1,n_points_to_sample=64):
        super().__init__()
        self.levels = levels
        self.n_mesh = n_mesh
        self.n_points_to_sample = n_points_to_sample
        # We initialize the source shape to be a sphere of radius 1
        src_mesh0 = ico_sphere(self.levels)
        self.src_mesh = Meshes(src_mesh0.verts_list()*self.n_mesh, src_mesh0.faces_list()*self.n_mesh)
        # import ipdb;ipdb.set_trace()
        self.deform_verts = torch.nn.Parameter(
            # torch.full((self.n_mesh,) + src_mesh0.verts_packed().shape, 0.0)
            0.1*torch.randn((self.n_mesh,) + src_mesh0.verts_packed().shape)
            )
        
        # self.register_buffer("src_mesh", src_mesh)
        pass
    def deform(self):
        
        new_src_mesh = self.src_mesh.offset_verts(self.deform_verts.flatten(start_dim=0,end_dim=1))
        return new_src_mesh.verts_list()
    def forward(self,ignore,n_noise = None,noise_mag = 0*0.1,n_points_to_sample=None):
        if n_points_to_sample is None:
            n_points_to_sample = self.n_points_to_sample
        # npp_hacks.save_every(self,tensor_to_numpy(self.deform_verts),'mesh.pkl')
        device = self.deform_verts.device
        self.src_mesh = self.src_mesh.to(self.deform_verts.device)
        
        # for mi in range(self.n_mesh):
        all_out_sample_src = []
        for ni in range(n_noise):
            noise = noise_mag*torch.randn_like(self.deform_verts)
            offset = (self.deform_verts + noise).flatten(start_dim=0,end_dim=1)
            new_src_mesh = self.src_mesh.offset_verts(offset)
            out_sample_src = sample_points_from_meshes(new_src_mesh, n_points_to_sample)
            all_out_sample_src.append(out_sample_src)
        all_out_sample_src = torch.cat(all_out_sample_src,dim=0)
       
        return all_out_sample_src
    def get_prior_losses(self):
        loss_edge = 0
        loss_normal = 0
        loss_laplacian = 0
        loss_face_area = 0 
        offset = (self.deform_verts).flatten(start_dim=0,end_dim=1)
        new_src_mesh = self.src_mesh.offset_verts(offset)
        loss_edge = loss_edge + mesh_edge_loss(new_src_mesh,order=2)
        loss_normal = loss_normal + mesh_normal_consistency(new_src_mesh)
        loss_laplacian = loss_laplacian + mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss_face_area = loss_face_area + - new_src_mesh.faces_areas_packed().mean()                

        losses = {  
            "edge": loss_edge,
            "normal": loss_normal,
            "laplacian": loss_laplacian,
            'area':loss_face_area,
        }
            
        return losses
    pass
