def StandardInverter():
    def __init__(self, setting_id,criterion):
        self.setting_id = setting_id
        if self.setting_id == 0:

            optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
            do_clip = True
            # optimizer = optim.Adam(list(inputs_F_c.parameters()) + list(inputs_F_f.parameters()), lr=lr)        
        elif self.setting_id == 1:
            optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
            do_clip = True                                        
        elif self.setting_id == 2:
            optimizer_other = optim.Adam([inputs_other], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)                
            # optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)
            do_clip = True
        
        self.lr_scheduler_other = lr_cosine_policy(self.lr, 100, iterations_per_layer)              
        self.criterion = criterion              
    def get_loss(self,outputs, targets):
        loss = self.criterion(outputs, targets).mean()
        
        return loss
    def get_auxiliary_loss(self,trends):
        loss_var_l1_first, loss_var_l2_first = get_image_prior_losses(inputs_for_prior_jit[:1])
        loss_var_l1_other, loss_var_l2_other = get_image_prior_losses(inputs_for_prior_jit)        
        trends['loss_var_l2_first'].append(loss_var_l2_first.item())