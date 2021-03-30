
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, batchnorm = False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        if batchnorm:
            self.regs = nn.ModuleList([nn.BatchNorm1d(dim_out, momentum=0.5) for dim_out in dims[1:]])
        else:
            self.regs = nn.ModuleList([nn.Identity() for dim_out in dims[1:]])

    def forward(self, x):
        # for l, conv in enumerate(self.convs):
        for l,layer in enumerate(self.layers):
            reg = self.regs[l]
            x = self.gate(reg(layer(x)))
        return x

class VanillaNet(nn.Module):
    def __init__(self, output_dim, body, use_dropout=False, dropout_p=0.5, feature_dim=None, n_ensambles = 1):
        # output_dim : output dimensions
        # body: nn.Module corresponding to body
        # n_ensambles: number of heads, default 1
        super(VanillaNet, self).__init__()
        self.n_ens = n_ensambles
        self.n_out = output_dim

        # patch to use body networks that have no feature dim attribute:
        if feature_dim is not None:
            body.feature_dim= feature_dim

        self.fc_head = layer_init(nn.Linear(body.feature_dim, self.n_out*self.n_ens))
        self.body = body
        if use_dropout:
            self.reg=nn.Dropout(p=dropout_p)
        else:
            self.reg =nn.Identity()

    def forward(self, x):
        phi = self.body(x)
        phi = self.reg(phi)
        y = self.fc_head(phi)
        if self.n_ens>1:
            y = y.reshape([*y.shape[:-1],  self.n_out, self.n_ens])
        return y
