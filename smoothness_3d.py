import torch
from pytorch3d.ops import knn_gather, knn_points
import inspect
import colorful
import numpy as np
import scipy.sparse as sp

#=======================================================
# def connected_components(adj_matrix):
#     """
#     Calculates the number of connected components in a graph given its adjacency matrix.
#     Assumes the adjacency matrix is a sparse tensor in CSR format.
#     """
#     num_vertices = adj_matrix.shape[0]
#     visited = torch.zeros(num_vertices, dtype=torch.bool)
#     components = []

#     def dfs(vertex):
#         visited[vertex] = True
#         components[-1].append(vertex)
#         for neighbor in adj_matrix.indices()[adj_matrix[vertex]:adj_matrix[vertex+1]]:
#             if not visited[neighbor]:
#                 dfs(neighbor)

#     for i in range(num_vertices):
#         if not visited[i]:
#             components.append([])
#             dfs(i)

#     return components

def connected_components(adj_matrix):
    num_components, component = sp.csgraph.connected_components((adj_matrix).to_dense().cpu().numpy())
    return num_components

def create_adjacency(knn):
    # Assuming you have a PyTorch tensor `knn` of shape (num_vertices, k),
    # where knn[i] contains the indices of the k nearest neighbors of vertex i.
    device = knn.device
    num_vertices, k = knn.shape[-2:]
    
    # Create the COO coordinates of the adjacency matrix
    row_indices = torch.arange(num_vertices,device=device).repeat_interleave(k)
    col_indices = knn.flatten()
    edge_indices = torch.stack([row_indices, col_indices], dim=0)

    # Create a sparse COO tensor
    adj_matrix = torch.sparse_coo_tensor(
        indices=edge_indices, 
        values=torch.ones_like(col_indices), 
        size=(num_vertices, num_vertices)
    )

    # Convert the COO tensor to CSR format for efficient matrix multiplication
    # adj_matrix = adj_matrix.coalesce().to('cuda').coo().to('cuda').csr()
    adj_matrix = adj_matrix.coalesce()
    return adj_matrix
#========================================================
def total_variation_3d_loss(points: torch.Tensor, colors: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    Computes the total variation loss for a point cloud.

    Args:
        points: A tensor of shape (batch_size, num_points, 3) representing the
            positions of the points in the point cloud.
        colors: A tensor of shape (batch_size, num_points, 3) representing the
            RGB color values of the points in the point cloud.
        k: The number of nearest neighbors to consider when computing the pairwise
            differences between neighboring points.

    Returns:
        The total variation loss for the point cloud.
    """
    trends = inspect.currentframe().f_back.f_locals['trends']
    
    # Compute the pairwise differences between neighboring points
    knn_obj = knn_points(points, points, K=k)
    if False:
        adj_matrix = create_adjacency(knn_obj.idx)
        ncc =connected_components(adj_matrix)
        trends['ncc'].append(len(ncc))
        import ipdb; ipdb.set_trace()
    neighbor_points = knn_gather(points, knn_obj.idx)
    neighbor_colors = knn_gather(colors, knn_obj.idx)
    diff_pos = neighbor_points - points[:,:,None]
    diff_col = neighbor_colors - colors[:,:,None]
    # import ipdb; ipdb.set_trace()
    if False:
        # Compute the norm of the differences
        norm_pos = torch.norm(diff_pos, dim=-1)
        norm_col = torch.norm(diff_col, dim=-1)
    else:
        norm_pos = diff_pos.abs().sum(dim=-1)
        norm_col = diff_col.abs().sum( dim=-1)        
    per_point_spread = (norm_pos < 1e-6).float().mean( dim=1)
    trends['per_point_spread'].append(per_point_spread.sum().item())
    # Compute the sum of the norms over all pairs of neighboring points
    # tv_loss = torch.sum(norm_pos) + torch.sum(norm_col)
    # tv_loss = torch.sum(points.norm(dim=1)) -torch.sum(norm_pos) + torch.sum(norm_col)
    # tv_loss = -torch.sum(norm_pos) + torch.sum(norm_col)
    # tv_loss = torch.sum(norm_pos) 
    # tv_loss = -torch.sum(norm_pos)
    tv_loss = torch.sum(norm_col)
    print(colorful.magenta("maximizing position variance, minimizing color variance"))
    return tv_loss
