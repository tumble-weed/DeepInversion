import torch
import torch.nn as nn
import torch.nn.init as init
import math
import modelnet_utils
tensor_to_numpy = lambda x: x.detach().cpu().numpy()
class TreeGCN(nn.Module):
    def __init__(self, batch, depth, features, degrees, support=10, node=1, upsample=False, activation=True):
        self.batch = batch
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth+1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation
        super(TreeGCN, self).__init__()

        self.W_root = nn.ModuleList([nn.Linear(features[inx], self.out_feature, bias=False) for inx in range(self.depth+1)])

        if self.upsample:
            self.W_branch = nn.Parameter(torch.FloatTensor(self.node, self.in_feature, self.degree*self.in_feature))
        
        self.W_loop = nn.Sequential(nn.Linear(self.in_feature, self.in_feature*support, bias=False),
                                    nn.Linear(self.in_feature*support, self.out_feature, bias=False))

        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.init_param()

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain('relu'))

        stdv = 1. / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        root = 0
        for inx in range(self.depth+1):
            root_num = tree[inx].size(1)
            repeat_num = int(self.node / root_num)
            root_node = self.W_root[inx](tree[inx])
            root = root + root_node.repeat(1,1,repeat_num).view(self.batch,-1,self.out_feature)

        branch = 0
        if self.upsample:
            branch = tree[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(self.batch,self.node*self.degree,self.in_feature)
            
            branch = self.W_loop(branch)

            branch = root.repeat(1,1,self.degree).view(self.batch,-1,self.out_feature) + branch
        else:
            branch = self.W_loop(tree[-1])

            branch = root + branch

        if self.activation:
            branch = self.leaky_relu(branch + self.bias.repeat(1,self.node,1))
        tree.append(branch)

        return tree

class Generator(nn.Module):
    def __init__(self, batch_size, features, degrees, support):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]
def test():
    DEGREE = [2,  2,  2,   2,      64]
    G_FEAT = [96, 64, 64,  64,    64, 3]
    batch_size = 2
    support = 10
    device = 'cuda'
    G = Generator(batch_size=batch_size, features=G_FEAT, degrees=DEGREE, support=support).to(device)
    z = 0.1*torch.randn(batch_size, 1, 96).to(device)
    tree = [z]
    fake_point = G(tree) 
    # fake_point = fake_point/fake_point.abs().max()
    fake_point = fake_point - fake_point.mean(dim=-1,keepdim=True)
    modelnet_utils.visualize_pointcloud(
        tensor_to_numpy(fake_point.permute(0,2,1)),savename = 'treegcn.png',c=None,AXIS_MAG=0.5)
    import ipdb;ipdb.set_trace()
    pass


if __name__ == '__main__':
    test()




