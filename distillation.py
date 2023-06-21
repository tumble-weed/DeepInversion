import torch
import colorful
import os
def create_input_for_distillation(masks_from_pr,zbuf):
    device = masks_from_pr.device
    input_masking_factor = 1
    if 'high_mask' in os.environ:
        input_masking_factor = 1e2
        # assert False
    DBG_HIGH_MASK = False
    if DBG_HIGH_MASK:
        print(colorful.gold('using high masking factor in DBG'))
        input_masking_factor = 1e2
    DBG_STE_INPUT_MASK = False
    if DBG_STE_INPUT_MASK:
        print(colorful.salmon("using STE for mask for input. actual gradient will be applied according to zbuf"))
        masks_for_input = (masks_from_pr - zbuf).detach() + zbuf
        masks_for_input = masks_for_input * input_masking_factor + ( 1 - input_masking_factor)*masks_for_input.detach()
    else:
        masks_for_input = masks_from_pr * input_masking_factor + ( 1 - input_masking_factor)*masks_from_pr.detach()                    

    # import ipdb; ipdb.set_trace()
    inputs0 = inputs
    inputs = inputs0 * masks_for_input + (masks_for_input == 0).float() * torch.randn(inputs.shape,device=device)
    inputs_for_prior = inputs0 * masks_for_input.detach() + (masks_for_input == 0).float() * torch.randn(inputs.shape,device=device)
    return inputs, inputs_for_prior
def get_distillation_losses(inputs_for_prior_jit,
                            inputs_from_pr,
                            masks_from_pr,
                            ENABLE_PR,
                            ENABLE_MESH,
                            trends):
    if ENABLE_PR == 1 or ENABLE_MESH == 1:
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
    else:
        # import ipdb; ipdb.set_trace()
        loss_l2_other = torch.norm(inputs_for_prior_jit.view(self.bs, -1), dim=1).mean()
            
    pass