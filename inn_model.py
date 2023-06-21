import torch
"""    
import FrEIA.framework as Ff
import FrEIA.modules as Fm
def subnet_fc(c_in, c_out):
    return torch.nn.Sequential(torch.nn.Linear(c_in, 512), torch.nn.ReLU(),
                        torch.nn.Linear(512,  c_out))

# def subnet_conv(c_in, c_out):
#     return torch.nn.Sequential(torch.nn.Conv2d(c_in, 256,   3, padding=1), torch.nn.ReLU(),
#                         torch.nn.Conv2d(256,  c_out, 3, padding=1))

# def subnet_conv_1x1(c_in, c_out):
#     return torch.nn.Sequential(torch.nn.Conv2d(c_in, 256,   1), torch.nn.ReLU(),
#                         torch.nn.Conv2d(256,  c_out, 1))


class INNModel(torch.nn.Module):
    def __init__(self):
        # from neural_point_prior import NPP
        super().__init__()
        self.inn = Ff.SequenceINN(3)
        for k in range(8):
            self.inn.append(Fm.AllInOneBlock, 
                       subnet_constructor=subnet_fc, permute_soft=True)
    def forward(self,sphere):
        vertsparam = self.inn(sphere)
        vertsparam = torch.tanh(vertsparam)
        return vertsparam
"""
import INN
class INNModel(torch.nn.Module):
    def __init__(self,dim=3):
        # from neural_point_prior import NPP
        super().__init__()
        D = 50
        self.model = INN.Sequential(
            INN.BatchNorm1d(3), 
            INN.Nonlinear(3, 'RealNVP'), 
            # INN.Linear(3,D),
            INN.JacobianLinear(3,D),
            # INN.JacobianLinear(3),
            INN.BatchNorm1d(D), 
            INN.Nonlinear(D, 'RealNVP'), 
            INN.JacobianLinear(D),
            INN.BatchNorm1d(D), 
            INN.Nonlinear(D, 'RealNVP'), 
            INN.JacobianLinear(D),
            INN.BatchNorm1d(D), 
            INN.Nonlinear(D, 'RealNVP'), 
            INN.JacobianLinear(D),
            INN.BatchNorm1d(D), 
            INN.Nonlinear(D, 'RealNVP'), 
            # INN.JacobianLinear(D), 
            INN.Linear(D,3), 
                    #    INN.ResizeFeatures(3, 1)
                       )

    def forward(self,sphere,extras={}):
        vertsparam0, logp, logdet  = self.model(sphere)
        vertsparam = torch.tanh(vertsparam0)
        extras.update(
            dict(
                logp = logp,
                logdet=logdet,
                vertsparam0 = vertsparam0
                
            )
            
        )
        return vertsparam
    

