import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.distributions.multivariate_normal import MultivariateNormal

import torchsummary
import pytorch_lightning as pl


def split(v):
    v1, v2 = torch.chunk(v, 2, dim=1)
    return v1, v2

def merge(v1, v2):
    return torch.cat([v1,v2], dim=1)

def perm_rand(n):
    return torch.eye(n)[torch.randperm(n)]

def perm_switch(dim):
    i = torch.cat([torch.arange(start=dim//2, end=dim), torch.arange(start=0, end=dim//2)])
    return torch.eye(dim)[i]

def perm_parity(dim):
    i = torch.cat([torch.arange(start=0, end=dim-1, step=2), torch.arange(start=1, end=dim, step=2)])
    return torch.eye(dim)[i]


class WeightNormedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_parameter('magnitude', nn.Parameter(data=torch.ones(1)))

    def forward(self, input):
        '''Weight Normalization (Salimans, 2016)'''
        return F.linear(input, self.magnitude*self.weight/self.weight.norm(), self.bias)


class CouplingMLP(nn.Module):
    def __init__(self, dim, width, depth, actf, weight_norm=False, dropout=False):
        super().__init__()
        self.depth = depth
        self.linear = WeightNormedLinear if weight_norm else nn.Linear

        self.actf = getattr(nn, actf)
        self.layers = [self.linear(dim, width), self.actf(), nn.Dropout(p=dropout)]
        for i in range(depth-1):
            self.layers.append(self.linear(width, width))
            self.layers.append(self.actf())
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(self.linear(width, dim * 2))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AffineCL(nn.Module):
    def __init__(self, data_dim, width, depth, actf, norm, drop, pm):
        super().__init__()
        self.coupling = CouplingMLP(data_dim // 2, width, depth, actf,
                                    weight_norm=norm, dropout=drop)
        self.register_buffer('perm', pm)
        self.register_parameter('ls_scale', nn.Parameter(data=torch.ones(1)))
        self.register_parameter('ls_shift', nn.Parameter(data=torch.zeros(1)))

    def encode(self, x):
        x = x @ self.perm
        x1, x2 = split(x)

        c = self.coupling(x1)
        s, t = split(c)
        s = torch.tanh(s) * self.ls_scale + self.ls_shift

        y1 = x1
        y2 = torch.exp(s) * x2 + t

        return merge(y1, y2), torch.sum(s, dim=1)

    def decode(self, y):
        y1, y2 = split(y)

        c = self.coupling(y1)
        s, t = split(c)
        s = torch.tanh(s) * self.ls_scale + self.ls_shift

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        x = merge(x1, x2)
        return x @ self.perm.T

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            z, _ = self.encode(x)
        return z

    def inverse(self, z):
        self.eval()
        with torch.no_grad():
            x = self.decode(z)
        return x


class Flow(pl.LightningModule):
    def __init__(self, data_dim, blocks, width, depth, actf, norm, perm,
                 lr, l2str, drop, bs, num_workers, train_ds, valid_ds, test_ds):
        super().__init__()
        # define the model
        if perm == 'reverse':
            self.chain = []
            for flow_step in range(blocks):
                self.chain.append(
                    AffineCL(data_dim, width, depth, actf, norm, drop,
                             perm_parity(data_dim) if flow_step == 0 else
                             perm_switch(data_dim)))
        elif perm == 'shuffle':
            self.chain = []
            for flow_step in range(blocks):
                self.chain.append(
                    AffineCL(data_dim, width, depth, actf, norm, drop,
                             perm_rand(data_dim)))
        self.chain = nn.ModuleList(self.chain)
        # define model prior
        self.register_buffer('prior_mean', torch.zeros(data_dim))
        self.register_buffer('prior_covariance', torch.eye(data_dim))
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        self.priors = dict() # for the case of multi-gpu training/inference
        # stored hparams
        self.msa = False
        self.norm = norm
        self.bs = bs
        self.lr = lr
        self.l2str = l2str
        # datasets
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.num_workers = num_workers

    def encode(self, x):
        z = x
        log_det = 0
        for acl in self.chain:
            z, log_det_inc = acl.encode(z)
            log_det += log_det_inc
        return z, log_det

    def decode(self, z):
        x = z
        for acl in self.chain[::-1]:
            x = acl.decode(x)
        return x

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            z, _ = self.encode(x)
        return z

    def inverse(self, z):
        self.eval()
        with torch.no_grad():
            x = self.decode(z)
        return x

    def log_prob_multigpu(self, z):
        def _prior_on_device(device, data_dim):
            return MultivariateNormal(
                torch.zeros(data_dim, device=device),
                torch.eye(data_dim, device=device))

        if str(z.device) not in self.priors:
            self.priors[str(z.device)] = _prior_on_device(z.device, z.shape[1])

        return self.priors[str(z.device)].log_prob(z)

    def cuda(self, device=None):
        super().cuda(device)
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        return self

    def nll(self, z, log_det):
        ll = self.prior.log_prob(z) if z.device == self.prior.loc.device else self.log_prob_multigpu(z)
        ll += log_det
        ll = torch.mean(ll) # mean over batch
        return -ll

    def l2(self):
        l2reg = [p[1]**2 for p in self.named_parameters()
                 if 'ls_scale' in p[0] or 'magnitude' in p[0]]
        if not self.norm:
            l2reg += [p[1].norm() for p in self.named_parameters() if 'weight' in p[0]]
        l2reg = torch.sum(torch.tensor(l2reg))
        return self.l2str * l2reg

    def loss(self, batch):
        x, _ = batch
        z, log_det = self.encode(x)
        return self.nll(z, log_det) + self.l2()

    def training_step(self, batch, batch_nb):
        loss = self.loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.loss(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_nb):
        loss = self.loss(batch)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

    # @pl.data_loader
    def train_dataloader(self):
        self.train_dl = DataLoader(self.train_ds, batch_size=self.bs, shuffle=True,
                                   num_workers=self.num_workers, pin_memory=True)
        return self.train_dl

    # @pl.data_loader
    def val_dataloader(self):
        self.valid_dl = DataLoader(self.valid_ds, batch_size=self.bs*2,
                                   num_workers=self.num_workers, pin_memory=True)
        return self.valid_dl

    # @pl.data_loader
    def test_dataloader(self):
        self.test_dl = DataLoader(self.test_ds, batch_size=self.bs*2,
                                   num_workers=self.num_workers, pin_memory=True)
        return self.test_dl
