import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FlowNMF3(nn.Module):
    '''DDSv3 or also DDS-C-MS (Constrained variant with Multi-Source modelling)
    using single class-conditional density model for all sources, as opposed to
    multiple models used by DDSv2 (DDS-C-SS) and DDSv1 (DDS-U-SS).
    '''
    def __init__(self, S, flow, n_comps_per_source,
                 nll_weight=0, nll_thresh=0, be_weight=0,
                 sm_weight=0, sp_frames=0, sp_ntsets=0,
                 Z_init='random', H_init='random', H_act=F.relu, H_detach=False,
                 optimizer=optim.Adam, lr=1e-2, sup='sum', log_det_J=False):
        super().__init__()
        self.register_buffer('S', S) # [data_dim, time]
        self.flow = flow
        self.n_sources = flow.hparams.num_classes
        self.n_cps = n_comps_per_source
        self.n_dim = self.S.shape[0]
        self.n_time = self.S.shape[1]

        assert self.flow.prior.event_shape[0] + self.n_sources == self.n_dim
        self.flow = self.flow.requires_grad_(False)

        # dictionary (semantic dimensions) # [n_sources * n_cps, n_sources]
        self.Z_s = torch.eye(self.n_sources).repeat_interleave(self.n_cps, dim=0).to(self.flow.device)

        # dictionary (nuisance dimensions) # [n_sources * n_cps, n_dim - n_sources]
        if Z_init == 'random':
            self.Z_n = self.flow.prior.sample((self.n_sources * self.n_cps,))
        elif Z_init == 'zero':
            self.Z_n = torch.zeros((self.n_sources * self.n_cps, self.n_dim - self.n_sources)).to(self.flow.device)
        else:
            raise NotImplementedError(f'Allowed options for Z_init are "random" and "zero".')

        # activations # [n_sources * n_cps, n_time]
        if H_init == 'random':
            self.H = torch.rand((self.n_sources * self.n_cps, self.n_time)) * torch.sqrt(S.mean() / (self.n_sources * self.n_cps))
        elif H_init == 'const':
            self.H = torch.ones((self.n_sources * self.n_cps, self.n_time)) / (self.n_sources * self.n_cps)
        else:
            raise NotImplementedError(f'Allowed options for H_init are "random" and "const".')
        self.H = self.H.to(self.flow.device)

        if H_detach:
            self.H_det = self.H.detach()

        # init for alternating update
        self.Z_s.requires_grad_(False) # semantic dims are held constant throughout the decomposition optimization
        self.Z_n.requires_grad_(True)
        self.H.requires_grad_(False)

        self.nll_weight = nll_weight
        self.nll_thresh = nll_thresh
        self.be_weight = be_weight
        self.sm_weight = sm_weight
        self.sp_frames = sp_frames # sparseness of frames (as in columns of H)
        self.sp_ntsets = sp_ntsets # sparseness of note [on/off]_sets / transitions
        self.H_act = H_act # F.relu, strelu_dangerous, strelu_mit_gurt, F.softplus
        self.H_detach = H_detach
        self.optimizer = optimizer([self.Z_n, self.H], lr=lr)
        self.lr = lr
        self.log_det_J = log_det_J

    def logprob_Z(self):
        logprob = self.flow.prior.log_prob(self.Z_n) # [n_sources * n_cps]

        if self.log_det_J:
            Z = torch.cat([self.Z_s, self.Z_n], dim=1) # [n_sources * n_cps, n_dim]
            X = self.flow.decode(Z)
            _, log_det = self.flow.encode(X) # [n_sources * n_cps]
            logprob += log_det # [n_sources * n_cps]

        nats_per_dim = logprob / self.flow.prior.event_shape[0] # / (n_dim - n_sources)
        return nats_per_dim # [n_sources * n_cps]

    def logprob_total(self, frame_wise_penalty_weight_norm=True):
        '''Weighted average of all components' logprobs (weighted by their activation entries in H)'''

        H = self.H_act(self.H_det if self.H_detach else self.H) # [n_sources * n_cps, n_time]
        if frame_wise_penalty_weight_norm:
            H /= (H.sum(dim=0, keepdim=True) + 1e-3) # [n_sources * n_cps, n_time] ... column-wise normalization
        lp_weights = H.sum(dim=1, keepdim=False) # [n_sources * n_cps]
        logprobs = self.logprob_Z() # [n_sources * n_cps]
        logprobs_weighted_sum = torch.sum(lp_weights * logprobs) # [n_sources * n_cps] -> [1]

        return logprobs_weighted_sum # [1]

    def forward(self):
        Z = torch.cat([self.Z_s, self.Z_n], dim=1) # [n_sources * n_cps, n_dim]
        W = F.relu(self.flow.decode(Z)).T # [n_dim, n_sources * n_cps]
        
        H = self.H_act(self.H) # [n_sources * n_cps, n_time]
        return W @ H # [n_dim, n_time]

    def diff(self):
        return torch.norm(self.S - self.forward())

    def cost(self):
        diff = self.diff()
        logprob = self.nll_weight and torch.sum(self.logprob_total()) * self.nll_weight

        return diff - logprob

    def cd_halfstep(self):
        self.cost().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # alternate updates
        self.Z_n.requires_grad_(not self.Z_n.requires_grad)
        self.H.requires_grad_(not self.H.requires_grad)

    def cd_step(self):
        self.cd_halfstep()
        self.cd_halfstep()

    def fit(self, max_iter=5000, tol=1e-15, tol_stag=10, tol_div=100, lr_reduc_fac=2):
        cost_best = cost_prev = self.cost().item()
        div = stag = 0 # counters of divergent / stagnating steps

        for step in range(max_iter):
            self.cd_step()

            with torch.no_grad():
                cost = self.cost()

            cost = cost.item()
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

        print(); print('Divergent.' if div > tol_div else 'Converged.')

    def get_X(self):
        with torch.no_grad():
            return self.forward().cpu().numpy()
    def get_LL_total(self):
        with torch.no_grad():
            return self.logprob_total().cpu().numpy()

    #self.Z_s.view(self.n_sources, self.n_cps, self.n_sources) # -> [source, component, 1-hot code]

    def get_comps_Z(self):
        Z = torch.cat([self.Z_s, self.Z_n], dim=1) # [n_sources * n_cps, n_dim]
        Z = Z.view(self.n_sources, self.n_cps, self.n_dim) # [n_sources, n_cps, n_dim]
        return {k: Z[k].T.cpu().detach().numpy() for k in range(self.n_sources)} # K x [n_dim, n_cps] for compatibility

    def get_comps_W(self):
        Z = torch.cat([self.Z_s, self.Z_n], dim=1) # [n_sources * n_cps, n_dim]
        W = F.relu(self.flow.inverse(Z)) # [n_sources * n_cps, n_dim]
        W = W.view(self.n_sources, self.n_cps, self.n_dim) # [n_sources, n_cps, n_dim]
        return {k: W[k].T.cpu().detach().numpy() for k in range(self.n_sources)} # K x [n_dim, n_cps] for compatibility

    def get_comps_H(self):
        with torch.no_grad():
            H = self.H_act(self.H) # [n_sources * n_cps, n_time]
        H = H.view(self.n_sources, self.n_cps, self.n_time) # [n_sources, n_cps, n_time]
        return {k: H[k].cpu().detach().numpy() for k in range(self.n_sources)} # K x [n_cps, n_time] for compatibility

    def get_comps_NPD(self):
        with torch.no_grad():
            NPD = self.logprob_Z() # [n_sources * n_cps]
        NPD = NPD.view(self.n_sources, self.n_cps) # [n_sources, n_cps]
        return {k: NPD[k].cpu().detach().numpy() for k in range(self.n_sources)} # K x [n_cps] for compatibility
