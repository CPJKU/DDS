import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FlowNMF(nn.Module):
    '''DDSv1 or also DDS-U-SS (Un-constrained variant with Sulti-Source modelling).'''
    def __init__(self, S, flows_dict,
                 nll_weight=0, nll_thresh=0, be_weight=0,
                 sm_weight=0, sp_frames=0, sp_ntsets=0,
                 Z_init='random', H_init='random', H_act=F.relu,
                 optimizer=optim.Adam, lr=1e-2, sup='sum'):
        super().__init__()
        self.register_buffer('S', S) # [data_dim, time]
        self.flows = flows_dict
        self.n_sources = len(flows_dict)
        self.n_dim = self.S.shape[0]
        self.n_time = self.S.shape[1]

        self.Z = {} # dictionary
        self.H = {} # activations
        for k in self.flows.keys():
            assert self.flows[k].prior.event_shape[0] == self.n_dim

            self.flows[k] = self.flows[k].requires_grad_(False)

            if Z_init == 'random':
                self.Z[k] = self.flows[k].prior.sample((self.n_time,)).T # [data_dim, time]
            elif Z_init == 'zero':
                self.Z[k] = torch.zeros_like(self.S, device=self.flows[k].device)
            elif Z_init == 'input':

                self.Z[k] = self.flows[k].forward(self.S.to(self.flows[k].device).T).T
                if self.n_sources > 1:
                    print('WARN: Z_init=input is not intended for n_components > 1')
            else:
                raise NotImplementedError(f'Allowed options for Z_init are random, zero and input.')

            if H_init == 'random':
                self.H[k] = torch.rand(self.n_time) * torch.sqrt(S.mean() / self.n_sources) # [time]
            elif H_init == 'const':
                self.H[k] = torch.Tensor([1. / self.n_sources] * self.n_time) # [time]
            else:
                raise NotImplementedError(f'Allowed options for H_init are random and const.')


            self.H[k] = self.H[k].to(self.Z[k].device) # put Hs where flows and Zs are

            # init for alternating update
            self.Z[k] = self.Z[k].requires_grad_(True)
            self.H[k] = self.H[k].requires_grad_(False)

        self.nll_weight = nll_weight
        self.nll_thresh = nll_thresh
        self.be_weight = be_weight
        self.sm_weight = sm_weight
        self.sp_frames = sp_frames # sparseness of frames (as in columns of H)
        self.sp_ntsets = sp_ntsets # sparseness of note [on/off]_sets / transitions
        self.H_act = H_act
        self.optimizer = optimizer(list(self.Z.values()) + list(self.H.values()), lr=lr)
        self.lr = lr

        superimpose = {
            'sum': lambda x: torch.sum(x, dim=0),
            'avg': lambda x: torch.mean(x, dim=0),
            'max': lambda x: torch.max(x, dim=0).values
        }
        superimpose['max+avg'] = lambda x: superimpose['max'](x) + superimpose['avg'](x)
        superimpose['sum-avg'] = lambda x: superimpose['sum'](x) - superimpose['avg'](x)
        self.superimpose = superimpose[sup]


    def logprob_Z(self, k):
        Z_k = self.Z[k].T
        logprob = self.flows[k].prior.log_prob(Z_k) # [time]
        nats_per_dim = logprob / self.n_dim

        return nats_per_dim

    def logprob_total(self):
        '''Weighted average of all components' logprobs (weighted by their activation entries in H)'''
        weighted_sum = torch.sum(torch.cat([(self.logprob_Z(k) * self.H_act(self.H[k])).unsqueeze(0) for k in self.flows.keys()], dim=0), dim=0)

        sum_of_weights = torch.sum(torch.cat([self.H_act(self.H[k]).unsqueeze(0) for k in self.flows.keys()], dim=0), dim=0)
        
        logprob_weighted_average = weighted_sum / (sum_of_weights + 1e-7) # prevent zero division
        return logprob_weighted_average

    def forward(self):
        components = []
        for k in self.flows.keys():
            W = F.relu(self.flows[k].decode(self.Z[k].T).T) # [n_dim, time]
            components.append((W * self.H_act(self.H[k])).unsqueeze(0))
        concatenated = torch.cat(components, dim=0)
        superimposed = self.superimpose(concatenated)
        return superimposed

    def diff(self):
        return torch.norm(self.S - self.forward())

    def cost(self):
        diff = self.diff()
        logprob = self.nll_weight and torch.sum(self.logprob_total()) * self.nll_weight

        if self.nll_thresh:
            logprob[logprob<-self.nll_thresh] = 0

        def binary_entropy(p):
            return - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
        def smoothness(H):
            return (H[:, :-1] - H[:, 1:])**2
        def hoyer_sparseness(x):
            '''Rainers code for Hoyer (2004) sparseness.'''
            sqn = torch.sqrt(torch.tensor(float(len(x))))
            l1 = torch.sum(torch.abs(x))
            l2 = torch.sqrt(torch.sum(torch.pow(x, 2)))
            return (sqn - (l1 / l2)) / (sqn - 1)

        if self.be_weight or self.sm_weight or self.sp_frames or self.sp_ntsets:
            H = torch.cat([self.H_act(self.H[k]).unsqueeze(0) for k in self.flows.keys()], dim=0)

        if self.sm_weight or self.sp_ntsets:
            temp_diff_H = smoothness(H)

        binary_entropy_H = self.be_weight and torch.sum(binary_entropy(H)) * self.be_weight
        smoothness_H = self.sm_weight and torch.sum(temp_diff_H) * self.sm_weight
        sparse_frames_H = self.sp_frames and torch.sum(torch.Tensor([hoyer_sparseness(H[:, i]) for i in range(H.shape[1])])) * self.sp_frames
        sparse_notesets_H = self.sp_ntsets and torch.sum(torch.Tensor([hoyer_sparseness(temp_diff_H[i, :]) for i in range(temp_diff_H.shape[0])])) * self.sp_ntsets

        return diff - logprob + binary_entropy_H + smoothness_H + sparse_frames_H + sparse_notesets_H

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
        return {k: F.relu(self.flows[k].inverse(self.Z[k].T).T).cpu().detach().numpy()
            for k in self.flows.keys()}

    def get_comps_H(self):
        with torch.no_grad():
            return {k: self.H_act(self.H[k]).cpu().numpy() for k in self.flows.keys()}

    def get_comps_NPD(self):
        with torch.no_grad():
            return {k: self.logprob_Z(k).cpu().numpy() for k in self.flows.keys()}
