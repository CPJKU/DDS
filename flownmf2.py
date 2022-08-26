import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FlowNMF2(nn.Module):
    '''DDSv2 or also DDS-C-SS (Constrained variant with Sulti-Source modelling).'''
    def __init__(self, S, flows_dict, n_comps_per_source,
                 nll_weight=0, nll_thresh=0, be_weight=0,
                 sm_weight=0, sp_frames=0, sp_ntsets=0,
                 Z_init='random', H_init='random', H_act=F.relu, H_detach=False,
                 optimizer=optim.Adam, lr=1e-2, sup='sum', log_det_J=False):
        super().__init__()
        self.register_buffer('S', S) # [data_dim, time]
        self.flows = flows_dict
        self.n_sources = len(flows_dict)
        self.n_cps = n_comps_per_source
        self.n_dim = self.S.shape[0]
        self.n_time = self.S.shape[1]

        self.Z = {} # dictionary
        self.H = {} # activations
        if H_detach:
            self.H_det = {}
        for k in self.flows.keys():
            assert self.flows[k].prior.event_shape[0] == self.n_dim

            self.flows[k] = self.flows[k].requires_grad_(False)

            if Z_init == 'random':
                self.Z[k] = self.flows[k].prior.sample((self.n_cps,)).T # [n_dim, n_cps]
            elif Z_init == 'zero':
                self.Z[k] = torch.zeros((self.n_dim, self.n_cps), device=self.flows[k].device)
            else:
                raise NotImplementedError(f'Allowed options for Z_init are random, zero and input.')

            if H_init == 'random':
                self.H[k] = torch.rand((self.n_cps, self.n_time)) * torch.sqrt(S.mean() / (self.n_sources * self.n_cps)) # [n_cps, time]
            elif H_init == 'const':
                self.H[k] = torch.Tensor(np.ones((self.n_cps, self.n_time)) / (self.n_sources * self.n_cps))
            else:
                raise NotImplementedError(f'Allowed options for H_init are random and const.')

            self.H[k] = self.H[k].to(self.Z[k].device) # put Hs where flows and Zs are

            if H_detach:
                self.H_det[k] = self.H[k].detach()

            # init for alternating update
            self.Z[k] = self.Z[k].requires_grad_(True)
            self.H[k] = self.H[k].requires_grad_(False)

        self.nll_weight = nll_weight
        self.nll_thresh = nll_thresh
        self.be_weight = be_weight
        self.sm_weight = sm_weight
        self.sp_frames = sp_frames # sparseness of frames (as in columns of H)
        self.sp_ntsets = sp_ntsets # sparseness of note [on/off]_sets / transitions
        self.H_act = H_act # F.relu
        self.H_detach = H_detach
        self.optimizer = optimizer(list(self.Z.values()) + list(self.H.values()), lr=lr)
        self.lr = lr
        self.log_det_J = log_det_J

    def logprob_Z(self, k):
        Z_k = self.Z[k].T # [n_cps, n_dim]
        logprob = self.flows[k].prior.log_prob(Z_k) # [n_cps]

        if self.log_det_J:
            X_k = self.flows[k].decode(Z_k)
            _, log_det = self.flows[k].encode(X_k) # [n_cps]
            logprob += log_det # [n_cps]

        nats_per_dim = logprob / self.n_dim
        return nats_per_dim # [n_cps]

    def logprob_total(self, frame_wise_penalty_weight_norm=True):
        '''Sum of weighted logprobs of all dict entries Z (weighted by activations in H).

        This is the updated variant where logprob penalties also aggregate additively across time.
        '''

        H = [self.H_act(self.H_det[k] if self.H_detach else self.H[k]) for k in self.flows.keys()] # n_sources x [n_cps, n_time]
        H = torch.cat(H, dim=0) # [n_sources * n_cps, n_time]
        if frame_wise_penalty_weight_norm:
            H /= (H.sum(dim=0, keepdim=True) + 1e-3) # [n_sources * n_cps, n_time] ... column-wise normalization
        lp_weights = H.sum(dim=1, keepdim=False) # [n_sources * n_cps]
        logprobs = torch.cat([self.logprob_Z(k) for k in self.flows.keys()], dim=0) # [n_sources * n_cps]
        logprobs_weighted_sum = torch.sum(lp_weights * logprobs) # [n_sources * n_cps] -> [1]

        return logprobs_weighted_sum # [1]

    def logprob_total_old_bad_normalization(self):
        '''Weighted average of all components' logprobs (weighted by their activation entries in H).
        However weights are averaged across time while reconstruction arror accumulates additively across frames,
        which for longer sequences (T >> 1) renders logprob term irrelevant!!!
        '''

        weighted_logprobs = []
        for k in self.flows.keys(): # [n_cps, time] --> [1, n_cps] --> [n_sources, n_cps]
            if self.H_detach:
                lp_k = torch.sum(self.H_act(self.H_det[k]), dim=1) # [n_cps, time] --> [n_cps]
            else:
                lp_k = torch.sum(self.H_act(self.H[k]), dim=1) # [n_cps, time] --> [n_cps]
            wlp_k = self.logprob_Z(k) * lp_k # [n_cps] * [n_cps] --> [n_cps]
            wlp_k = wlp_k.unsqueeze(0) # [n_cps] --> [1, n_cps]
            weighted_logprobs.append(wlp_k) # n_sources * [1, n_cps]
        weighted_logprobs = torch.cat(weighted_logprobs, dim=0) # n_sources * [1, n_cps] --> [n_sources, n_cps]
        weighted_sum = torch.sum(weighted_logprobs) # [n_sources, n_cps] -> [1]

        weights = torch.cat([torch.sum(self.H_act(self.H[k]), dim=1).unsqueeze(0) \
                             for k in self.flows.keys()], dim=0) # [n_sources, n_cps]
        sum_of_weights = torch.sum(weights) # [n_sources, n_cps] -> [1]
        
        logprob_weighted_average = weighted_sum / (sum_of_weights + 1e-7) # prevent zero division
        return logprob_weighted_average # [1]

    def forward(self):
        W = torch.cat([F.relu(self.flows[k].decode(self.Z[k].T).T) for k in self.flows.keys()], dim=1) # [n_dim, n_sources * n_cps]
        H = torch.cat([self.H_act(self.H[k]) for k in self.flows.keys()], dim=0) # [n_sources * n_cps, n_time]
        return W @ H

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
        for k in self.flows.keys():
            self.Z[k].requires_grad_(not self.Z[k].requires_grad)
            self.H[k].requires_grad_(not self.H[k].requires_grad)

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

    def get_comps_Z(self):
        return {k: self.Z[k].cpu().detach().numpy() for k in self.flows.keys()}

    def get_comps_W(self):
        return {k: F.relu(self.flows[k].inverse(self.Z[k].T).T).cpu().detach().numpy() for k in self.flows.keys()}

    def get_comps_H(self):
        with torch.no_grad():
            return {k: self.H_act(self.H[k]).cpu().numpy() for k in self.flows.keys()}

    def get_comps_NPD(self):
        with torch.no_grad():
            return {k: self.logprob_Z(k).cpu().numpy() for k in self.flows.keys()}
