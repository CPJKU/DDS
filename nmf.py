import numpy as np
import matplotlib.pyplot as plt
import librosa.display

import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.collections import QuadMesh


class PyTorchNMF(nn.Module):
    '''NMF or also oc-NMF (Over-complete Non-negative Matrix Factorization).'''
    def __init__(self, S, n_components, W_init=None, H_init=None, W_norm=False,
                 optimizer=optim.Adam, lr=1e-2, clip=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.S = torch.Tensor(S).to(self.device)
        self.lr = lr
        self.clip = clip
        self.W_norm = W_norm

        # 'random' init (as in sklearn.decomposition.NMF)
        scale = np.sqrt(S.mean().item() / n_components)

        if W_init is not None:
            W = W_init
        else:
            W = np.random.rand(S.shape[0], n_components) * scale

        if H_init is not None:
            H = np.full((n_components, S.shape[1]), H_init / n_components)
        else:
            H = np.random.rand(n_components, S.shape[1]) * scale

        self.W = torch.Tensor(W).to(self.device)
        self.H = torch.Tensor(H).to(self.device)
        self.optimizer = optimizer([self.W, self.H], lr=lr)

    def forward(self):
        return torch.nan_to_num(self.W / self.W.norm(dim=0)) @ self.H if self.W_norm else self.W @ self.H

    def cost(self):
        return torch.norm(self.S - self.forward())

    def clip_to_bounds(self):
        with torch.no_grad():
            self.W[self.W<0] = 0
            self.H[self.H<0] = 0
            if self.clip:
                self.W[self.W>self.clip] = self.clip
                self.H[self.H>self.clip] = self.clip

    def cd_halfstep(self):
        self.forward()
        self.cost().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.clip_to_bounds()

    def cd_step(self, W_fixed):
        self.W.requires_grad_(False)
        self.H.requires_grad_(True)
        self.cd_halfstep()
        if not W_fixed:
            self.W.requires_grad_(True)
            self.H.requires_grad_(False)
            self.cd_halfstep()

    def fit(self, max_iter=5000, tol=1e-15, tol_stag=10, tol_div=100, lr_reduc_fac=2, W_fixed=False):
        cost_best = cost_prev = self.cost().item()
        div = stag = 0 # counters of divergent / stagnating steps

        self.W.requires_grad_(True)
        self.H.requires_grad_(False)

        for step in range(max_iter):
            self.cd_step(W_fixed)
            cost = self.cost().item()
            print(f'{step} {self.lr} {cost}', end='\r')
            div = div * (cost >= cost_prev) + (cost >= cost_prev)
            stag = stag * (cost > cost_best) + (cost > cost_best)
            if stag > tol_stag:
                for g in self.optimizer.param_groups:
                    g['lr'] /= lr_reduc_fac
                self.lr /= lr_reduc_fac
                stag = 0
            if np.abs(cost - cost_prev) < tol or div > tol_div:
                break
            cost_prev = cost
            cost_best = cost if cost < cost_best else cost_best

        print(); print(f'Divergent.' if div > tol_div else f'Converged. {np.abs(cost - cost_prev)}')

        return self.getW(), self.getH()

    def getW(self):
        return self.W.detach().cpu().numpy()

    def getH(self):
        return self.H.detach().cpu().numpy()
