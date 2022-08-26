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


class CouplingSqueezingCNN(nn.Module):
    '''Squeezing ConvNet built purely out of 1D convolutions.
    Squeezing is achieved via stride. No Max Pooling used.
    '''
    def __init__(self, dim, kernel_size, stride, actf, weight_norm=False):
        super().__init__()
        self.linear = WeightNormedLinear if weight_norm else nn.Linear
        assert kernel_size >= stride

        self.actf = getattr(nn, actf)

        channels = 1
        dimension = dim

        self.conv_layers = list()

        while self._dim_change(dimension, kernel_size, stride) > 0:
            self.conv_layers.append(
                nn.Conv1d(channels, channels*stride, kernel_size, stride))
            channels *= stride
            dimension = self._dim_change(dimension, kernel_size, stride)

            self.conv_layers.append(self.actf())

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.linear_layer = self.linear(channels * dimension, dim * 2)

    def _dim_change(self, dim_in, kernel_size, stride, padding=0, dilation=1):
        '''Change in the Length dimension induced by torch.nn.Conv1D

        Source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''

        dividend = dim_in + 2 * padding - dilation * (kernel_size - 1) - 1
        dim_out = int((dividend / stride) + 1)
        return dim_out

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension for convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, x.shape[1] * x.shape[2]) # remove channel dimension
        x = self.linear_layer(x)
        return x


class CouplingReceptiveCNN(nn.Module):
    def __init__(self, dim, kernel_size, channels, depth, actf, dilated=False, weight_norm=False):
        super().__init__()
        assert kernel_size > 1 and channels > 1 and depth > 0
        assert kernel_size % 2 == 1, 'even kernel size disturbs the output shape preservation'

        self.actf = getattr(nn, actf)

        self.conv_layers = [nn.Conv1d(1, channels, kernel_size, stride=1, dilation=1,
                                      padding=kernel_size//2, padding_mode='circular'),
                            self.actf()]
        for i in range(depth-1):
            dilation = kernel_size ** (i+1) if dilated else 1
            padding = ((kernel_size + (kernel_size-1) * (dilation-1)) if dilated else kernel_size)//2
            self.conv_layers.append(
                nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=dilation,
                          padding=padding, padding_mode='circular'))
            self.conv_layers.append(self.actf())

        self.conv_layers.append(
            nn.Conv1d(channels, 2, 1 if dilated else kernel_size, stride=1, dilation=1,
                      padding=0 if dilated else kernel_size//2, padding_mode='circular'))

        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension for convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        return x


class AffineCL(nn.Module):
    def __init__(self, hp, data_dim, width, depth, actf, norm, drop):
        super().__init__()

        if hp.acl_arch == 'mlp':
            self.coupling = CouplingMLP(
                hp.data_dim // 2, hp.mlp_width, hp.mlp_depth, hp.actf, weight_norm=hp.norm, dropout=hp.drop)
        elif hp.acl_arch == 'cnn_sqz':
            self.coupling = CouplingSqueezingCNN(
                hp.data_dim // 2, hp.cnn_kernel_size, hp.cnn_stride, hp.actf, weight_norm=hp.norm)
        elif hp.acl_arch == 'cnn_rec':
            self.coupling = CouplingReceptiveCNN(
                hp.data_dim // 2, hp.cnn_kernel_size, hp.cnn_channels, hp.cnn_depth, hp.actf,
                dilated=hp.cnn_dilated, weight_norm=hp.norm)
        else:
            raise NotImplementedError(f'Architecture {self.hparams.acl_arch} is not a valid option.')

        self.register_parameter('ls_scale', nn.Parameter(data=torch.ones(1)))
        self.register_parameter('ls_shift', nn.Parameter(data=torch.zeros(1)))

    def encode(self, x):
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
        return x


class ActNorm(nn.Module):
    '''
    Based on: https://github.com/asteroidhouse/INN-exploding-inverses

    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output has zero mean and unit variance per dimension.

    After initialization, `bias` and `logs` will be trained as parameters.
    '''

    def __init__(self, data_dim, scale=1., max_scale=0, actnorm_eps=0):
        super().__init__()
        # register mean and scale
        size = [1, data_dim]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = data_dim
        self.scale = scale
        self.inited = False
        self.stable_eps = actnorm_eps
        self.max_scale = max_scale

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = - torch.mean(input.clone(), dim=0, keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=0, keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, x, inverse=False):
        if inverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, inverse=False):
        if self.max_scale == 0:
            scale = torch.exp(self.logs) + self.stable_eps
        else:
            scale = torch.exp(np.log(self.max_scale) * torch.tanh(self.logs))
        self.last_scale = scale

        if inverse:
            x = x / scale
        else:
            x = x * scale

        logdet = torch.sum(torch.log(scale))

        if inverse:
            logdet *= -1

        return x, logdet

    def encode(self, x):
        y = self._center(x, inverse=False)
        y, logdet = self._scale(y, inverse=False)
        return y, logdet

    def decode(self, y):
        x, logdet = self._scale(y, inverse=True)
        x = self._center(x, inverse=True)
        return x, logdet

    def forward(self, input, inverse=False):
        if not self.inited:
            self.initialize_parameters(input)

        if inverse:
            output, _ = self.decode(input)
        else:
            output, _ = self.encode(input)

        return output


class Invertible1x1AsDenseLayer(nn.Module):
    '''
    Based on: https://github.com/asteroidhouse/INN-exploding-inverses

    Invertible 1x1 Convolution for 1-dimensional data: invertible Dense layer.
    '''
    def __init__(self, data_dim, LU_decomposed=False, from_log_s=False):
        super().__init__()
        w_shape = [data_dim, data_dim]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.W = nn.Parameter(torch.Tensor(w_init))
        else:
            P, L, U = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(U) # extracts the diagonal of U into s
            U = torch.triu(U, 1) # sets diagonal of U to 0s

            self.register_buffer('P', P)
            self.L = nn.Parameter(L)
            self.U = nn.Parameter(U)

            if from_log_s:
                self.register_buffer('sign_s', torch.sign(s))
                self.log_s = nn.Parameter(torch.log(torch.abs(s)))
            else:
                self.s = nn.Parameter(s)

            self.l_mask = torch.tril(torch.ones(*w_shape), -1)
            self.eye = torch.eye(*w_shape)

        self.LU_decomposed = LU_decomposed
        self.from_log_s = from_log_s

    def get_weight(self, device=None, inverse=False):
        if not self.LU_decomposed:
            logdet = torch.log(torch.abs(torch.det(self.W))) #torch.slogdet(self.W)[1]
            if inverse:
                W = torch.inverse(self.W)
            else:
                W = self.W
        else:
            self.l_mask = self.l_mask.to(device)
            self.eye = self.eye.to(device)

            L = self.L * self.l_mask + self.eye # fix diag(L) to 1s beyond optimization
            U = self.U * self.l_mask.T.contiguous() # fix diag(U) to 0s beyond optimization

            if self.from_log_s:
                s = self.sign_s * torch.exp(self.log_s)
                logdet = torch.sum(self.log_s)
            else:
                s = self.s
                logdet = torch.sum(torch.log(torch.abs(s)))

            U += torch.diag(s) # put s on the diagonal of U

            if inverse:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
                P_inv = torch.inverse(self.P)

                W = U_inv @ (L_inv @ P_inv)
            else:
                W = self.P @ (L @ U)

        return W, logdet

    def encode(self, x):
        W, logdet = self.get_weight(device=x.device, inverse=False)
        y = F.linear(x, W, bias=None)
        return y, logdet

    def decode(self, y):
        W, _ = self.get_weight(device=y.device, inverse=True)
        x = F.linear(y, W, bias=None)
        return x

    def forward(self, input, inverse=False):
        if inverse:
            output = self.decode(input)
        else:
            output, _ = self.encode(input)
        return output


class FixedPermutation(nn.Module):
    def __init__(self, permutation_matrix):
        super().__init__()
        self.register_buffer('permutation_matrix', permutation_matrix)

    def encode(self, x):
        return x @ self.permutation_matrix

    def decode(self, y):
        return y @ self.permutation_matrix.T


class FlowStep(nn.Module):
    def __init__(self, step_index, hp):
        #data_dim, width, depth, actf, norm, drop, perm_type, use_actnorm, LU_decomposed, from_log_s
        super().__init__()
        self.perm_type = hp.perm_type
        self.use_actnorm = hp.use_actnorm

        self.coupling_layer = AffineCL(hp, hp.data_dim, hp.mlp_width, hp.mlp_depth, hp.actf, hp.norm, hp.drop)

        if self.perm_type == 'reverse':
            pm = perm_parity(hp.data_dim) if step_index == 0 else perm_switch(hp.data_dim)
            self.mixdims = FixedPermutation(pm)
        elif self.perm_type == 'shuffle':
            pm = perm_rand(hp.data_dim)
            self.mixdims = FixedPermutation(pm)
        elif self.perm_type == '1x1':
            self.mixdims = Invertible1x1AsDenseLayer(
                hp.data_dim, LU_decomposed=hp.LU_decomposed, from_log_s=hp.from_log_s)
        else:
            raise ValueError(f'Permutation type `{perm_type}` is not an option.')

        if self.use_actnorm:
            self.actnorm = ActNorm(hp.data_dim)

    def encode(self, x):
        log_det_inc = 0

        if self.use_actnorm:
            x, log_det = self.actnorm.encode(x)
            log_det_inc += log_det

        if self.perm_type == '1x1':
            x, log_det = self.mixdims.encode(x)
            log_det_inc += log_det
        else:
            x = self.mixdims.encode(x)

        y, log_det = self.coupling_layer.encode(x)
        log_det += log_det_inc # log_det is of shape [batch_size] while log_det_inc is a scalar

        return y, log_det

    def decode(self, y):
        x = self.coupling_layer.decode(y)
        x = self.mixdims.decode(x)
        if self.use_actnorm:
            x, _ = self.actnorm.decode(x)

        return x


class Glow(pl.LightningModule):
    def __init__(self, data_dim, blocks, acl_arch, mlp_width, mlp_depth,
                 cnn_kernel_size, cnn_stride, cnn_channels, cnn_depth, cnn_dilated,
                 actf, drop, norm, l2str, lr,
                 perm_type='1x1', use_actnorm=True, LU_decomposed=True, from_log_s=True):
        super().__init__()
        # store all *args and **kwargs of __init__ into self.hparams namespace
        self.save_hyperparameters()
        # define the model
        self.chain = []
        for step_index in range(blocks):
            flow_step = FlowStep(step_index, self.hparams)
            self.chain.append(flow_step)
        self.chain = nn.ModuleList(self.chain)
        # define model prior
        self.register_buffer('prior_mean', torch.zeros(data_dim))
        self.register_buffer('prior_covariance', torch.eye(data_dim))
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        self.priors = dict() # for the case of multi-gpu training/inference

    def encode(self, x):
        z = x
        log_det = 0
        for flow_step in self.chain:
            z, log_det_inc = flow_step.encode(z)
            log_det += log_det_inc
        return z, log_det

    def decode(self, z):
        x = z
        for flow_step in self.chain[::-1]:
            x = flow_step.decode(x)
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

    # SECTION START: to ensure that samples from prior are on the same device as the model
    def cuda(self, device=None):
        super().cuda(device=device)
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        return self

    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        return self
    # SECTION END: to ensure that samples from prior are on the same device as the model

    def nll(self, z, log_det):
        ll = self.prior.log_prob(z) if z.device == self.prior.loc.device else self.log_prob_multigpu(z)
        ll += log_det
        ll = torch.mean(ll) # mean over batch
        return -ll

    def l2(self):
        l2reg = [p[1]**2 for p in self.named_parameters()
                 if 'ls_scale' in p[0] or 'magnitude' in p[0]]
        if not self.hparams.norm:
            l2reg += [p[1].norm() for p in self.named_parameters() if 'weight' in p[0]]

        l2reg += [p[1].norm() for p in self.named_parameters() if 'weight' in p[0] and 'conv' in p[0]]

        l2reg = torch.sum(torch.tensor(l2reg))
        return self.hparams.l2str * l2reg

    def loss(self, batch):
        x, _ = batch
        z, log_det = self.encode(x)
        nll = self.nll(z, log_det)
        l2 = self.l2()
        log_det = torch.mean(log_det).detach() # mean over batch, detach for logging
        return nll + l2, nll, l2, log_det

    def training_step(self, batch, batch_idx):
        loss, nll, l2, log_det = self.loss(batch)
        self.log('train_loss', loss, logger=False, on_step=False, on_epoch=True)
        return {'loss': loss, 'l2': l2, 'log_det': log_det}

    def validation_step(self, batch, batch_idx):
        loss, nll, l2, log_det = self.loss(batch)
        self.log('val_loss', loss, logger=False, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_log_det': log_det}

    def test_step(self, batch, batch_idx):
        loss, nll, l2, log_det = self.loss(batch)
        self.log('test_loss', loss, logger=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return self.optimizer


class ConditionalGlow(pl.LightningModule):
    def __init__(self, num_classes, data_dim, blocks, acl_arch, mlp_width, mlp_depth,
                 cnn_kernel_size, cnn_stride, cnn_channels, cnn_depth, cnn_dilated,
                 actf, drop, norm, l2str, lr,
                 perm_type='1x1', use_actnorm=True, LU_decomposed=True, from_log_s=True,
                 nc_width=0, sem_mse=100, nui_ce=0.05):
        super().__init__()
        assert num_classes < data_dim

        self.save_hyperparameters()

        self.chain = []
        for step_index in range(blocks):
            flow_step = FlowStep(step_index, self.hparams)
            self.chain.append(flow_step)
        self.chain = nn.ModuleList(self.chain)

        nc_dim = nc_width or (data_dim // 2)
        self.nuisance_classifier = nn.Sequential(
            nn.Linear(data_dim - num_classes, nc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(nc_dim, nc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(nc_dim, num_classes)
        )

        # to make this trainable, 'register_parameter' instead of '_buffer'
        self.register_buffer('prior_mean', torch.zeros(data_dim - num_classes))
        self.register_buffer('prior_covariance', torch.eye(data_dim - num_classes))
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        self.priors = dict()

        self.ce_loss = nn.CrossEntropyLoss() # needs scalar labels (dtype = long)
        self.mse_loss = nn.MSELoss() # needs 1-hot labels (dtype = float)

        self.automatic_optimization = False # self.manual_backward(obj) instead of obj.backward()

    def ice_loss(self, batch):
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes=self.hparams.num_classes).float()
        y_shuffled = y[torch.randperm(y.size()[0])]

        z, log_det = self.encode(x)
        zs, zn = torch.split(z, [self.hparams.num_classes, self.hparams.data_dim-self.hparams.num_classes], dim=1)

        # 1) minimize NLL (for stability)
        nll = self.nll(zn, log_det)

        # 2) minimize MSE on the 1-hot class ID
        # Our deviation from the original approach using cross-entropy on softmax(logits):
        # we need the dimensions to get as close as possible to the 1-hot class code (softmax cross-entropy doesn't grant that)
        semantic_error = self.mse_loss(zs, y_one_hot)

        # 3) maximize CE on nuisance classifier logits s.t. nuisance dims are less informative for nuisance classifier in the future
        logits_nc = self.nuisance_classifier(zn)
        cross_entropy_nc_zn = self.ce_loss(logits_nc, y_shuffled)

        # 4) add L2 regularization penalty
        l2 = self.l2()

        # full objective
        obj_main = nll + self.hparams.sem_mse * semantic_error - 1 * self.hparams.nui_ce * cross_entropy_nc_zn + l2

        return obj_main, semantic_error, cross_entropy_nc_zn, nll, log_det, l2

    def training_step(self, batch, batch_idx):
        optim_main, optim_nc = self.optimizers()

        x, y = batch

        z, log_det = self.encode(x)
        zn = z[:, self.hparams.num_classes:]

        # train the nuisance classifier to predict labels well from nuisance dimensions
        zn_train = zn.clone().detach()
        for i in range(5):
            logits_nc = self.nuisance_classifier(zn_train)
            loss_nc = self.ce_loss(logits_nc, y)
            obj_nc = torch.mean(loss_nc)
            optim_nc.zero_grad()
            self.manual_backward(obj_nc)
            torch.nn.utils.clip_grad_norm_(self.nuisance_classifier.parameters(), 1e+6)
            optim_nc.step()

        # compute the independence Cross Entropy (iCE) objective to update the Normalizing Flow model
        obj_main, semantic_error, cross_entropy_nc_zn, nll, log_det, l2 = self.ice_loss(batch)

        # update the main model unsing the iCE objective
        optim_main.zero_grad()
        self.manual_backward(obj_main)
        torch.nn.utils.clip_grad_norm_(self.chain.parameters(), 1e+6)
        optim_main.step()

        # extract quantities for logging
        nc_loss_last = obj_nc.detach()
        mse_zs = torch.mean(semantic_error).detach()
        neg_ce_nc_zn = torch.mean(- cross_entropy_nc_zn).detach()
        log_det = torch.mean(log_det).detach() # mean over batch, detach for logging

        self.log('train_loss', obj_main, prog_bar=True, logger=False, on_step=False, on_epoch=True)

        return {'loss': obj_main.detach(), 'nll': nll.detach(), 'l2': l2.detach(), 'log_det': log_det.detach(),
                'mse_zs': mse_zs, 'neg_ce_nc_zn': neg_ce_nc_zn, 'nc_loss_last': nc_loss_last}

    def validation_step(self, batch, batch_idx):
        loss, *_ = self.ice_loss(batch)
        self.log('val_loss', loss, logger=False, on_step=False, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss, *_ = self.ice_loss(batch)
        self.log('test_loss', loss, logger=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        self.optimizer_main = optim.Adam(self.chain.parameters(), lr=self.hparams.lr)
        self.optimizer_nc = optim.Adam(self.nuisance_classifier.parameters(), lr=self.hparams.lr)
        return self.optimizer_main, self.optimizer_nc

    # -------------- All of the code below is duplicate of Glow code above --------------
    # attempt to reuse this code via inheritance and skipping of __init__() on Glow level caused
    # trouble with lightning module initialization for training

    def encode(self, x):
        z = x
        log_det = 0
        for flow_step in self.chain:
            z, log_det_inc = flow_step.encode(z)
            log_det += log_det_inc
        return z, log_det

    def decode(self, z):
        x = z
        for flow_step in self.chain[::-1]:
            x = flow_step.decode(x)
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

    # SECTION START: to ensure that samples from prior are on the same device as the model
    def cuda(self, device=None):
        super().cuda(device=device)
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        return self

    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.prior = MultivariateNormal(self.prior_mean, self.prior_covariance)
        return self
    # SECTION END: to ensure that samples from prior are on the same device as the model

    def nll(self, z, log_det):
        ll = self.prior.log_prob(z) if z.device == self.prior.loc.device else self.log_prob_multigpu(z)
        ll += log_det
        ll = torch.mean(ll) # mean over batch
        return -ll

    def l2(self):
        l2reg = [p[1]**2 for p in self.named_parameters()
                 if 'ls_scale' in p[0] or 'magnitude' in p[0]]
        if not self.hparams.norm:
            l2reg += [p[1].norm() for p in self.named_parameters() if 'weight' in p[0]]

        l2reg += [p[1].norm() for p in self.named_parameters() if 'weight' in p[0] and 'conv' in p[0]]

        l2reg = torch.sum(torch.tensor(l2reg))
        return self.hparams.l2str * l2reg
