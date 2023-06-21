import torch
import dutils
import os
import numpy as np
tensor_to_numpy = lambda t: t.detach().cpu().numpy()
dutils.SAVE_DIR = '/root/evaluate-saliency-4/fong-invert/DeepInversion/debugging'
def get_circle(n_ticks,filled=False,mode='uniform',device='cpu'):
    if mode == 'uniform':
        theta = np.linspace(0, 2 * np.pi, n_ticks)
        if filled:
            r = np.linspace(0, 1, theta.shape[0])
        else:
            r = 1
    elif mode == 'random':
        theta = np.random.uniform(0, 2 * np.pi, n_ticks)
        if filled:
            r = np.random.uniform(0, 1, theta.shape[0])
        else:
            r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    circle = torch.tensor(np.vstack((x, y)).T).float().to(device)
    return circle
def get_sphere(n_ticks,filled=False,mode='uniform',device='cpu'):
    if mode == 'uniform':
        theta = np.linspace(0, np.pi, n_ticks)
        phi = np.linspace(0, 2 * np.pi, n_ticks)
        theta, phi = np.meshgrid(theta, phi)
        theta, phi = theta.flatten(), phi.flatten()
        if filled:
            r = np.linspace(0, 1, theta.shape[0])
        else:
            r = 1
    elif mode == 'random':
        theta = np.random.uniform(0, np.pi, n_ticks*n_ticks)
        phi = np.random.uniform(0, 2 * np.pi, n_ticks*n_ticks)
        if filled:
            r = np.random.uniform(0, 1, theta.shape[0])
        else:
            r = 1
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    sphere = torch.tensor(np.vstack((x, y, z)).T).float().to(device)
    return sphere   


def eye_init(self,i,l):
    # for the first layer, make groups of eye
    d0 = l.weight.data.shape[1]
    d1 = l.weight.data.shape[0]
    if i == 0:
        l.weight.data = torch.eye(d0).repeat(d1//d0,1)
    # for all layers after the first, use eye init
    if i > 0 and i < self.N - 1:
        l.weight.data = torch.eye(d0)
class FCResidualBlock(torch.nn.Module):
    def __init__(self,D,Dsmall,non_lin=None):
        super().__init__()
        self.bottleneck = torch.nn.Linear(D,Dsmall)
        self.non_lin = non_lin
        self.expand = torch.nn.Linear(Dsmall,D)
    def forward(self,x):
        y = self.bottleneck(x)
        if self.non_lin is not None:
            y = self.non_lin(y)
        y = self.expand(y)
        return x + y
        
    
class NPP(torch.nn.Module):
    def __init__(self,inD=2,D=100,N=3,outD=None,max_d=1,non_lin=None,init_mode='eye'):
        super().__init__()
        self.inD = inD
        self.max_d = max_d
        if non_lin is None:
            non_lin = torch.nn.functional.selu
        self.non_lin = non_lin
        if outD is None:
            outD = inD
        self.outD = outD
        self.D = D
        self.N = 3
        self.layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        for i in range(self.N):
            if i == 0:
                d0 = self.inD
            else:
                d0 = self.D
            if i <  self.N - 1:
                d1 = self.D
            else:
                d1 = self.outD

            l = torch.nn.Linear(d0,d1,bias=True)
            # """
            if init_mode == 'eye':
                eye_init(self,i,l)
            # """
            """
            if (i < self.N - 1) and (i>0):
                l = FCResidualBlock(d1,d1//2,non_lin=non_lin)
            else:
                l = torch.nn.Linear(d0,d1,bias=True)
            """
            self.layers.append(l)
            self.layer_norms.append(torch.nn.LayerNorm(d1))
        pass
    def forward(self,x):
        # """
        M = x.max().detach()
        m = x.min().detach()
        x0 = x
        x = 2*(x - m)/(M-m) -1
        flag_left = (x < 0).float()
        flag_right = 1- flag_left
        eps = 1e-2
        x = torch.log( (x + 1 + eps)*flag_left + flag_right )   - torch.log( (1 - x + eps)*flag_right + flag_left )            
        # """            

        # import ipdb;ipdb.set_trace()
        for i,l in enumerate(self.layers):
            x = l(x)
            # x = self.layer_norms[i](x)
            
            if i < self.N - 1:
                x = self.non_lin(x)
                pass

            else:
                x = torch.tanh(x)
                """
                max_x = x.abs().max().detach()
                max_x = max(max_x,1)
                x = x/max_x
                """
                x = torch.cat([x[:,:self.inD]* self.max_d ,x[:,self.inD:]],dim=-1)
                
        # import ipdb;ipdb.set_trace()
        return x
    
    pass
def test_2d():
    nruns = 10
    for i in range(nruns):
        nticks = 100
        Y,X = torch.meshgrid(torch.linspace(-1,1,nticks),torch.linspace(-1,1,nticks))
        Y,X = Y.flatten(),X.flatten()
        x = torch.stack([X,Y],dim=-1)
        assert x.ndim == 2
        net = NPP(inD=2,D=100,N=3)
        trf_x = net(x)
        trf_x = tensor_to_numpy(trf_x)
        assert trf_x.shape == x.shape
        # plot trf_x
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(trf_x[:,0],trf_x[:,1])
        plt.draw()
        fname = os.path.join(dutils.SAVE_DIR,f'trf_x_{i}.png')
        print(fname)
        plt.savefig(fname)
        plt.close()
        del net    
def test_2d_colored():
    nruns = 10
    for i in range(nruns):
        nticks = 100
        Y,X = torch.meshgrid(torch.linspace(-1,1,nticks),torch.linspace(-1,1,nticks))
        Y,X = Y.flatten(),X.flatten()
        x = torch.stack([X,Y],dim=-1)
        black = torch.zeros(x.shape[0],3)
        x = torch.cat([x,black],dim=-1)
        assert x.ndim == 2
        net = NPP(inD=2+3,D=100,N=3)
        trf_x = net(x)
        trf_x = tensor_to_numpy(trf_x)
        assert trf_x.shape == x.shape
        # plot trf_x
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(trf_x[:,0],trf_x[:,1],c=(trf_x[:,2:]+1)/2)
        plt.draw()
        fname = os.path.join(dutils.SAVE_DIR,f'trf_x_{i}.png')
        print(fname)
        plt.savefig(fname)
        plt.close()
        del net    

def test_2d_colored_output():
    nruns = 10
    for i in range(nruns):
        nticks = 100
        Y,X = torch.meshgrid(torch.linspace(-1,1,nticks),torch.linspace(-1,1,nticks))
        Y,X = Y.flatten(),X.flatten()
        x = torch.stack([X,Y],dim=-1)
        # black = torch.zeros(x.shape[0],3)
        # x = torch.cat([x,black],dim=-1)
        assert x.ndim == 2
        net = NPP(inD=2,outD=5,D=100,N=3)
        trf_x = net(x)
        trf_x = tensor_to_numpy(trf_x)
        # assert trf_x.shape == x.shape
        # plot trf_x
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(trf_x[:,0],trf_x[:,1],c=(trf_x[:,2:]+1)/2)
        plt.draw()
        fname = os.path.join(dutils.SAVE_DIR,f'trf_x_{i}.png')
        print(fname)
        plt.savefig(fname)
        plt.close()
        del net            
        
def test_3d_slices():
    net = NPP(inD=3,outD=3,D=100,N=3)
    nticks = 100
    Y,X = torch.meshgrid(torch.linspace(-1,1,nticks),torch.linspace(-1,1,nticks))
    Y,X = Y.flatten(),X.flatten()
    x = torch.stack([X,Y],dim=-1)
    nruns = 10
    z = torch.linspace(-1,-1 + 0.1,nruns)
    for i in range(nruns):
        assert x.ndim == 2
        x_and_z = torch.cat([x,z[i]*torch.ones(x.shape[0],1)],dim=-1)
        trf_x = net(x_and_z)
        trf_x = tensor_to_numpy(trf_x)
        # assert trf_x.shape == x.shape
        # plot trf_x
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(trf_x[:,0],trf_x[:,1])
        plt.draw()
        fname = os.path.join(dutils.SAVE_DIR,f'trf_x_{i}.png')
        print(fname)
        plt.savefig(fname)
        plt.close()
        # del net                    
def test_3d():
    net = NPP(inD=3,outD=3,D=100,N=3)
    nticks = 25
    # 3d meshgrid
    Z,Y,X = torch.meshgrid(torch.linspace(-1,1,nticks),torch.linspace(-1,1,nticks),torch.linspace(-1,1,nticks))
    Z,Y,X = Z.flatten(),Y.flatten(),X.flatten()
    x = torch.stack([X,Y,Z],dim=-1)
    assert x.ndim == 2
    
    trf_x = net(x)
    trf_x = tensor_to_numpy(trf_x)
    # assert trf_x.shape == x.shape
    # plot trf_x
    import matplotlib.pyplot as plt
    # 3d plot
    from mpl_toolkits.mplot3d import Axes3D
    #view from multiple angles
    n_angles = 10
    for i in range(n_angles):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(trf_x[:,0],trf_x[:,1],trf_x[:,2])
        a = i*360/n_angles
        ax.view_init(azim=a)
        plt.draw()
        fname = os.path.join(dutils.SAVE_DIR,f'3d_trf_x_azim_{a}.png')
        print(fname)
        plt.savefig(fname)
        plt.close()
    for i in range(n_angles):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(trf_x[:,0],trf_x[:,1],trf_x[:,2])
        e = -30 + i*60/n_angles
        ax.view_init(elev=e)
        plt.draw()
        fname = os.path.join(dutils.SAVE_DIR,f'3d_trf_x_elev_{e}.png')
        print(fname)
        plt.savefig(fname)
        plt.close()        
    # del net                    

def test_mnist():
    import torchvision
    import skimage
    import numpy as np

    device = 'cuda'
    data = torchvision.datasets.MNIST(root='.',train=False,download=True)
    # get 1 sample
    sample_ix = -1
    for x,y in data:
        if y == 5:
            sample_ix += 1
            if sample_ix == 2:
                break
    x = np.array(x)
    assert x.shape == (28,28)
    # get a contour of x
    if x.max() > 1:
        x = x/255
    print(x.max())
    # x = x > 0.5
    x_contours = skimage.measure.find_contours(x,0.5)
    assert len(x_contours) == 1
    x_contour = x_contours[0]
    x_contour_im = np.zeros_like(x)
    for contour in x_contours:
        x_contour_im[contour[:,0].astype(int),contour[:,1].astype(int)] = 1
    # assert False
    dutils.img_save(x_contour_im,'contours.png')
    dutils.img_save(x,'x.png')
    # get coordinates of the contour
    v,u = x_contour[:,1],x_contour[:,0]
    v = v - v.mean()
    u = u - u.mean()
    v = v/(x.shape[1]/2)
    u = u/(x.shape[0]/2)
    u = torch.from_numpy(u).float().to(device)
    v = torch.from_numpy(v).float().to(device)
    npp_model = NPP(inD=2,outD=2,D=100,N=10,non_lin=torch.nn.functional.relu)
    optimizer = torch.optim.Adam(npp_model.parameters(),lr=1e-4)
    npp_model.to(device)
    n_ticks = 10000
    sphere = get_circle(n_ticks,filled=False,mode='uniform',device=device)
    print(sphere.shape)
    
    from collections import defaultdict
    trends = defaultdict(list)
    def chamfer_loss_2d(pred_u,pred_v,u,v):
        ix_of_nearest_gt = torch.argmin(torch.cdist(torch.stack([u,v],dim=-1),torch.stack([pred_u,pred_v],dim=-1),p=2),dim=0)
        ix_of_nearest_pred = torch.argmin(torch.cdist(torch.stack([pred_u,pred_v],dim=-1),torch.stack([u,v],dim=-1),p=2),dim=0)
        loss1 = ((torch.stack([u[ix_of_nearest_gt],v[ix_of_nearest_gt]],dim=-1) - torch.stack([pred_u,pred_v],dim=-1))**2).mean()
        loss2 = ((torch.stack([pred_u[ix_of_nearest_pred],pred_v[ix_of_nearest_pred]],dim=-1) - torch.stack([u,v],dim=-1))**2).mean()
        return 0*loss1 + loss2
        
              
    for i in range(1000):
        print(i)
        pred = npp_model(sphere)
        pred_u,pred_v = pred[:,0],pred[:,1]
        # chamfer loss w.r.t u and v
        loss = chamfer_loss_2d(pred_u,pred_v,u,v)
        optimizer.zero_grad()
        if i > 500:
            optimizer.param_groups[0]['lr'] = 1e-5
        loss.backward()
        trends['loss'].append(loss.item())
        optimizer.step()
    
    x_pred = np.zeros_like(x)
    x_pred[((pred_u + 1)*(x.shape[0]/2)).detach().cpu().numpy().astype(int),((pred_v+1)*(x.shape[1]/2)).detach().cpu().numpy().astype(int)] = np.linspace(0,1,n_ticks)        
    dutils.img_save(x_pred,'x_pred.png')
    dutils.save_plot2(trends['loss'],'contour_loss','contour_loss.png')
    # print(trends['loss'])
    pass
if __name__ == '__main__':
    # test_2d()
    # test_2d_colored_output()
    # test_3d_slices()
    # test_3d()
    test_mnist()