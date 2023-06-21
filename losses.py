import torch
from torch import nn
def get_verifier_loss(inputs_jit,output,net_student,
                      adi_scale,detach_student):
    # R_ADI
    kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
    loss_verifier_cig = torch.zeros(1)
    if adi_scale!=0.0:
        if detach_student:
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
    return loss_verifier_cig
"""
get_bn_loss(loss_r_feature_layers,self.first_bn_multiplier,self.adi_scale,self.detach_student,net_student)
"""
def get_bn_loss(loss_r_feature_layers,first_bn_multiplier):
    device = loss_r_feature_layers[0].device
    kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
    rescale = [first_bn_multiplier] + [ (1) for _ in range(len(loss_r_feature_layers)-1)]

    loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])
    return loss_r_feature


def get_3d_losses(pr_model,trends):
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
    loss_l2_vertsparam = (vertsparam_for_l2_vertsparam - vertsparam_for_l2_vertsparam.mean(dim=0,keepdim=True)).norm(dim=1,p=2).sum()
    # import ipdb; ipdb.set_trace()
    trends['loss_l2_vertsparam'].append(loss_l2_vertsparam.item())
    L2_VERTS = float(os.environ.get('L2_VERTS', 0))

    # hack(locals(),L2_VERTS = 0)
    from smoothness_3d import total_variation_3d_loss
    tv_3d = total_variation_3d_loss(pr_model.vertsparam[None], pr_model.sh_param[None], k = 6)
    trends['tv_3d'].append(tv_3d.item())
    return mesh_volume,loss_l2_vertsparam,tv_3d