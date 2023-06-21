import torch
import npp_hacks
from mesh_model import MeshModel   
import inn_model 
tensor_to_numpy = lambda x: x.detach().cpu().numpy()



class RecursiveNPP(torch.nn.Module):
    def __init__(self):
        from neural_point_prior import NPP
        super().__init__()
        self.model = NPP(inD=3,D=99,N=10,outD=3,max_d=1,
            # non_lin=torch.nn.functional.leaky_relu
            )
        self.layers= self.model.layers
    def forward(self,sphere):
        last_sphere = sphere
        N = 100
        change = 0
        for ni in range(N):
            point_cloud_6d = self.model(last_sphere)

            """
            sphere_norms = last_sphere.norm(2,dim=-1,keepdim=True)
            sphere_norms = sphere_norms + (sphere_norms==0).float()
            sphere_unit = sphere/sphere_norms  
            delta = point_cloud_6d[:,:1]
            delta = (delta + 1)/2
            # delta = (delta + 1)
            # delta = (delta **10/(2**10))
            
            delta = -delta 
            new_norm = (sphere_norms + 0.4*delta)
            vertsparam =  new_norm.clip(0,1) * sphere_unit
            # vertsparam = (point_cloud_6d[:,:3])
            """
            vertsparam = (point_cloud_6d[:,:3])*1./N + last_sphere#*0.9
            # vertsparam = (point_cloud_6d[:,:3])*0.3 + last_sphere*0.7
            change = (last_sphere - vertsparam).norm(2,dim=-1).mean()
            last_sphere = vertsparam
            if change < 1e-3:
                break
        return vertsparam

class CompositeNPP(torch.nn.Module):
    def __init__(self,device):
        from neural_point_prior import NPP
        super().__init__()
        self.npp_models = torch.nn.ModuleList()
        for ni in range(5):
            modeli = NPP(inD=3,D=513,N=2,outD=3,max_d=1,
            # non_lin=torch.nn.functional.leaky_relu,
            non_lin=torch.nn.functional.sigmoid,
            init_mode = 'random',
            ).to(device)
            self.npp_models.append(modeli)
    def forward(self,sampled_sphere):
        n_models = len(self.npp_models)
        vertsparam = 0
        for ni in range(n_models):
            point_cloud_6d = self.npp_models[ni](sampled_sphere[0].permute(1,0))
            vertsparam += point_cloud_6d[:,:3]
        vertsparam = vertsparam/n_models                                     
        return vertsparam


class SpectralModel(torch.nn.Module):
    def __init__(self, n_modes, out_dims,n_chan,mag=1,device='cuda'):
        super().__init__()
        from torch_harmonics import InverseRealSHT
        from torch import nn
        self.n_chan = n_chan
        self.coeffs = nn.Parameter(mag*torch.randn(n_chan,n_modes, n_modes+1, dtype=torch.complex128,device=device))
        # __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        self.ishts = [InverseRealSHT(
                                out_dims[0], out_dims[1],
                                    #n_modes, 2*n_modes, 
                                #h_out=out_dims[0], w_out=out_dims[1], 
                                mmax=n_modes+1,lmax = n_modes,
                                grid="equiangular").to(device) for _ in range(n_chan)]

    def forward(self,ignore):
        out = []
        for ci,isht in enumerate(self.ishts):
            
            out.append(isht(self.coeffs[ci] +0.001*torch.randn_like(self.coeffs[ci])).float())
        # return self.isht(self.coeffs)
        out_3d = torch.tanh(torch.stack(out,dim=0))
        out = out_3d.permute(1,2,0).reshape(-1,self.n_chan)
        return out

class WrappedPRModel(torch.nn.Module):
    def __init__(self, npoints,n_mesh=1):
        super().__init__()
        self.n_mesh = n_mesh
        self.vertparam = torch.nn.Parameter(torch.randn(self.n_mesh,npoints,3))
    def forward(self,ignore):
        return self.vertparam

class PointCloudGenerator(torch.nn.Module):
    def get_norm_(self):
        self.norms = torch.nn.ModuleList()
        self.norms.append(torch.nn.InstanceNorm1d(self.D))
        # self.norms.append(torch.nn.BatchNorm1d(self.D))
        for ni in range(self.N-1):              
            self.norms.append(torch.nn.InstanceNorm1d(self.D))
            # self.norms.append(torch.nn.BatchNorm1d(self.D))
        pass
    def apply_norm_(self,x,ni):
        if self.__dict__.get('norms',None) is not None:
            if ni < len(self.layers)-1:
                x = self.norms[ni](x)
        return x
    def __init__(self,npoints,inD,N,D):
        super().__init__()
        self.npoints = npoints
        self.inD = inD
        self.D = D
        self.N = N
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.inD,self.D))
        for ni in range(self.N-1):              
            self.layers.append(torch.nn.Linear(self.D,self.D))
        self.get_norm_()
        self.layers.append(torch.nn.Linear(self.D,self.npoints*3))
        
    def forward(self,x):
        # x = torch.randn(self.npoints,1,device='cuda')
        for ni,layer in enumerate(self.layers):
            x = layer(x)
            # if ni < len(self.layers)-1:
            #     x = self.norms[ni](x)
            x = self.apply_norm_(x,ni)
            if ni < len(self.layers)-1:
                x = torch.nn.functional.leaky_relu(x)
        # x = torch.tanh(x)
        x = x.view(x.shape[0],-1,3)
        return x


def get_point_model(device,ns,gt=None,n_mesh=1):
    POINTCLOUD_TYPE0 = os.environ.get('POINTCLOUD_TYPE0',None)
    if POINTCLOUD_TYPE0 == 'npp' or False:
        os.environ['POINTCLOUD_TYPE'] = 'npp'
        from neural_point_prior import NPP
        pointcloud_model =  NPP(inD=3,D=255,N=10,outD=3,max_d=1,
        non_lin=torch.nn.functional.leaky_relu,
        init_mode = 'random'
        ).to(device)
        scale_logit = torch.nn.Parameter(torch.tensor(0.,device=device))
    elif False:
        os.environ['POINTCLOUD_TYPE'] = 'sh'
        # from npp_hacks import SpectralModel
        n_ticks = 32
        #1,4,9,16
        pointcloud_model = SpectralModel( 16, (n_ticks,n_ticks),3,mag=0.1,device=device).to(device)
    elif False:
        os.environ['POINTCLOUD_TYPE'] = 'generator'
        ns['USE_GENERATOR'] = True
        from npp_hacks import PointCloudGenerator
        pointcloud_model = PointCloudGenerator(1024,100,3,100).to(device)
        pointcloud_model.train()
    elif False:      
        os.environ['POINTCLOUD_TYPE'] = 'pr'      
        # from npp_hacks import WrappedPRModel
        n_ticks = 32
        pointcloud_model = WrappedPRModel( 64,n_mesh=n_mesh).to(device)            
        vertparam = pointcloud_model.vertparam.detach().clone()
        # vertsparam = pr_model.vertsparam.detach().clone()
        assert vertparam.shape[-1] == 3
        lengths = vertparam.norm(dim=-1)[...,None]
        lengths[lengths<1] = 1
        pointcloud_model.vertparam.data.copy_(
            # vertsparam.data.clamp(-1,1)
            vertparam/lengths
        )
        assert pointcloud_model.vertparam.norm(dim=-1).max() <= 1.0001        
    elif True:      
        os.environ['POINTCLOUD_TYPE'] = 'inn'      
        n_ticks = 32
        pointcloud_model = inn_model.INNModel().to(device)   
    elif False:      
        os.environ['POINTCLOUD_TYPE'] = 'mesh'      
        n_ticks = 32
        pointcloud_model = MeshModel(n_mesh=n_mesh).to(device)           
    elif False:
        # assert False      
        os.environ['POINTCLOUD_TYPE'] = 'tree'      
        from tree_gcn import Generator
        # n_ticks = 32
        # pointcloud_model = MeshModel(n_mesh=n_mesh).to(device)           
        DEGREE = [2,  2,  2,   2,      64]
        G_FEAT = [96, 64, 64,  64,  64,   3]
        batch_size = 40
        support = 10        
        pointcloud_model = Generator(batch_size=batch_size, features=G_FEAT, degrees=DEGREE, support=support).to(device)
    else:
        # from npp_hacks import CompositeNPP
        pointcloud_model = CompositeNPP(device)    
    return pointcloud_model