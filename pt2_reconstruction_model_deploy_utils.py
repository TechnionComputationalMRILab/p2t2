import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List


import numpy as np
import math

from epg_parallel_batch import epg_parallel_batch


class FC_Model(nn.Module):

    def __init__(self, input_channel, channels, output_channel, apply_softmax=True):
        """
        :param input_channel: 32 = n_echos
        :param channels: [256]*6
        :param output_channel: 120 = args.T2_log.num_samples (original = 60)
        """
        super().__init__()
        self.in_channel = input_channel
        self.channels = channels
        self.out_channel = output_channel
        self.apply_softmax = apply_softmax
        self.net = self._create_net()

    def _create_net(self):
        input_channel = self.in_channel
        channels = self.channels
        output_channel = self.out_channel

        n_layers = len(channels)
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(in_features=input_channel, out_features=channels[i], bias=True))
            else:
                layers.append(nn.Linear(in_features=channels[i - 1], out_features=channels[i], bias=True))
            # TODO: ask Moti - tf.keras.layers.LeakyReLU() default alpha= 0.3, torch default=0.01!
            layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        layers.append(nn.Linear(in_features=channels[-1], out_features=output_channel, bias=False))
        if self.apply_softmax:
            layers.append(nn.Softmax(dim=-1))

        model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        output = self.net(x)

        return output


class MLP_Model(nn.Module):

    def __init__(self, input_channel, channels, output_channel):
        """
        :param input_channel: 32 = n_echos
        :param channels: [256]*6
        :param output_channel: 120 = args.T2_log.num_samples (original = 60)
        """
        super().__init__()
        self.in_channel = input_channel
        self.channels = channels
        self.out_channel = output_channel

        self.net = self._create_net()

    def _create_net(self):
        input_channel = self.in_channel
        channels = self.channels
        output_channel = self.out_channel

        n_layers = len(channels)
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(in_features=input_channel, out_features=channels[i], bias=True))
            else:
                layers.append(nn.Linear(in_features=channels[i - 1], out_features=channels[i], bias=True))
            # TODO: ask Moti - tf.keras.layers.LeakyReLU() default alpha= 0.3, torch default=0.01!
            layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        layers.append(nn.Linear(in_features=channels[-1], out_features=output_channel, bias=False))
        layers.append(nn.Softmax(dim=-1))

        model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        output = self.net(x)
        return output


class EPG_Signal():
    def __init__(self, NT2grid:int = 60, T2_min:float=10.0, T2_max:float=2000.0, device='cpu') -> None:  # nEchoes:int,
        
        # self.n = nEchoes
        
        self.T2_log = torch.logspace(
            math.log10(T2_min), 
            math.log10(T2_max), 
            steps=NT2grid, 
            base=10.0, requires_grad=True, device=device, dtype=torch.float32)
        self.TR = 1.e10


        self.rad = math.pi / 180.0
        # self.alpha_exc =  torch.FloatTensor([90. * self.rad])

    def __call__(self, tau_vec, alpha_vec):
        """
        alpha_vec: [batch, 1, numTE] - degree 
        tau_vec:   [batch, 1] - only the TE_min
        """
        # nRates = self.R2.shape[0]
        # batch = tau_vec.shape[0]
        # epg_dict = torch.zeros((batch, self.n, nRates), dtype=torch.float64)
        
        batch_size = tau_vec.shape[0]
        T2 = self.T2_log.repeat(batch_size,1)
        T1 = 1000.0 * torch.ones_like(T2)

        epg_dict = epg_parallel_batch(
            angles_rad=alpha_vec*self.rad, 
            TE=tau_vec, 
            TR=self.TR, 
            T1=T1, 
            T2=T2, 
            B1=1.
            )  # T2 - in sec? 

        # for i, (alpha,tau) in enumerate(zip(alpha_vec, tau_vec)):

        #     alpha_rad = alpha * self.rad
        #     epg_dict[i,...] = self.epg_signal(tau=tau, alpha=alpha_rad)

        return epg_dict

    def epg_signal(self, tau, alpha):
        nRates = self.R2.shape[0]
        tau = tau / 2.0
        # defining signal matrix
        H = torch.zeros((self.n, nRates), dtype=torch.float64)

        # RF mixing matrix
        T = self.fill_T(self.n, alpha)

        # Selection matrix to move all traverse states up one coherence level
        S = self.fill_S(self.n)

        for iRate in range(nRates):
            # Relaxation matrix
            R2 = self.R2[iRate]
            R1 = self.R1[iRate]

            R0 = torch.zeros((3, 3), dtype=torch.float64)
            R0[0, 0] = torch.exp(-tau * R2)
            R0[1, 1] = torch.exp(-tau * R2)
            R0[2, 2] = torch.exp(-tau * R1)

            R = self.fill_R(self.n, tau, R0, R2)
            # Precession and relaxation matrix
            P = torch.mm(R, S)
            # Matrix representing the inter-echo duration
            E = torch.mm(torch.mm(P, T), P)
            H = self.fill_H(R, self.n, E, H, iRate, self.alpha_exc)
            # end
        return H

    def fill_S(self, n):
        the_size = 3 * n + 1
        S = torch.zeros((the_size, the_size), dtype=torch.float64)
        S[0, 2] = 1.0
        S[1, 0] = 1.0
        S[2, 5] = 1.0
        S[3, 3] = 1.0
        for o in range(2, n + 1):
            offset1 = ((o - 1) - 1) * 3 + 2
            offset2 = ((o + 1) - 1) * 3 + 3
            if offset1 <= (3 * n + 1):
                S[3 * o - 2, offset1 - 1] = 1.0  # F_k <- F_{k-1}
            # end
            if offset2 <= (3 * n + 1):
                S[3 * o - 1, offset2 - 1] = 1.0  # F_-k <- F_{-k-1}
            # end
            S[3 * o, 3 * o] = 1.0  # Z_order
        # end for
        return S

    def fill_T(self, n, alpha):
        T0 = torch.zeros((3, 3), dtype=torch.float64)
        # T0[0, :] = [torch.cos(alpha / 2.0) ** 2, torch.sin(alpha / 2.0) ** 2, torch.sin(alpha)]
        T0[0, 0] = torch.cos(alpha / 2.0) ** 2
        T0[0, 1] = torch.sin(alpha / 2.0) ** 2
        T0[0, 2] = torch.sin(alpha)
        
        # T0[1, :] = [torch.sin(alpha / 2.0) ** 2, torch.cos(alpha / 2.0) ** 2, -torch.sin(alpha)]
        T0[1, 0] = torch.sin(alpha / 2.0) ** 2
        T0[1, 1] = torch.cos(alpha / 2.0) ** 2
        T0[1, 2] = -torch.sin(alpha)

        # T0[2, :] = [-0.5 * torch.sin(alpha), 0.5 * torch.sin(alpha), torch.cos(alpha)]
        T0[2, 0] = -0.5 * torch.sin(alpha)
        T0[2, 1] = 0.5 * torch.sin(alpha)
        T0[2, 2] = torch.cos(alpha)


        T = torch.zeros((3 * n + 1, 3 * n + 1), dtype=torch.float64)
        T[0, 0] = 1.0
        T[1:3 + 1, 1:3 + 1] = T0
        for itn in range(n - 1):
            T[(itn + 1) * 3 + 1:(itn + 2) * 3 + 1, (itn + 1) * 3 + 1:(itn + 2) * 3 + 1] = T0
            
        return T

    def fill_R(self, n, tau, R0, R2):
        R = torch.zeros((3 * n + 1, 3 * n + 1), dtype=torch.float64)
        R[0, 0] = torch.exp(-tau * R2)
        R[1:3 + 1, 1:3 + 1] = R0
        for itn in range(n - 1):
            R[(itn + 1) * 3 + 1:(itn + 2) * 3 + 1, (itn + 1) * 3 + 1:(itn + 2) * 3 + 1] = R0
        # end
        return R

    def fill_H(self, R, n, E, H, iRate, alpha_exc):
        x = torch.zeros((R.shape[0], 1), dtype=torch.float64)
        x[0] = torch.sin(alpha_exc)
        x[1] = 0.0
        x[2] = torch.cos(alpha_exc)
        for iEcho in range(n):
            x = torch.mm(E, x)
            H[iEcho, iRate] = x[0]
        # end for IEcho
        return H


class EPG_Generator():
    def __init__(self, nEchoes, TEmin, fa_list, T2grid_type='linear', t2_range=None):
        self.nEchoes = nEchoes
        self.TEmin = TEmin
        self.fa_list = fa_list
        self.T2grid_type = T2grid_type
        self.t2_range = t2_range


    def epg_signal(self, n, tau, R1vec, R2vec, alpha, alpha_exc):
        nRates = R2vec.shape[0]
        tau = tau / 2.0
        # defining signal matrix
        H = np.zeros((n, nRates))
        # RF mixing matrix
        T = self.fill_T(n, alpha)
        # Selection matrix to move all traverse states up one coherence level
        S = self.fill_S(n)

        for iRate in range(nRates):
            # Relaxation matrix
            R2 = R2vec[iRate]
            R1 = R1vec[iRate]

            R0 = np.zeros((3, 3))
            R0[0, 0] = np.exp(-tau * R2)
            R0[1, 1] = np.exp(-tau * R2)
            R0[2, 2] = np.exp(-tau * R1)

            R = self.fill_R(n, tau, R0, R2)
            # Precession and relaxation matrix
            P = np.dot(R, S)
            # Matrix representing the inter-echo duration
            E = np.dot(np.dot(P, T), P)
            H = self.fill_H(R, n, E, H, iRate, alpha_exc)
        return H

    def fill_S(self, n):
        the_size = 3 * n + 1
        S = np.zeros((the_size, the_size))
        S[0, 2] = 1.0
        S[1, 0] = 1.0
        S[2, 5] = 1.0
        S[3, 3] = 1.0
        for o in range(2, n + 1):
            offset1 = ((o - 1) - 1) * 3 + 2
            offset2 = ((o + 1) - 1) * 3 + 3
            if offset1 <= (3 * n + 1):
                S[3 * o - 2, offset1 - 1] = 1.0  # F_k <- F_{k-1}
            if offset2 <= (3 * n + 1):
                S[3 * o - 1, offset2 - 1] = 1.0  # F_-k <- F_{-k-1}
            S[3 * o, 3 * o] = 1.0  # Z_order
        return S

    def fill_T(self, n, alpha):
        T0 = np.zeros((3, 3))
        T0[0, :] = [math.cos(alpha / 2.0) ** 2, math.sin(alpha / 2.0) ** 2, math.sin(alpha)]
        T0[1, :] = [math.sin(alpha / 2.0) ** 2, math.cos(alpha / 2.0) ** 2, -math.sin(alpha)]
        T0[2, :] = [-0.5 * math.sin(alpha), 0.5 * math.sin(alpha), math.cos(alpha)]

        T = np.zeros((3 * n + 1, 3 * n + 1))
        T[0, 0] = 1.0
        T[1:3 + 1, 1:3 + 1] = T0
        for itn in range(n - 1):
            T[(itn + 1) * 3 + 1:(itn + 2) * 3 + 1, (itn + 1) * 3 + 1:(itn + 2) * 3 + 1] = T0
        return T

    def fill_R(self, n, tau, R0, R2):
        R = np.zeros((3 * n + 1, 3 * n + 1))
        R[0, 0] = np.exp(-tau * R2)
        R[1:3 + 1, 1:3 + 1] = R0
        for itn in range(n - 1):
            R[(itn + 1) * 3 + 1:(itn + 2) * 3 + 1, (itn + 1) * 3 + 1:(itn + 2) * 3 + 1] = R0
        return R

    def fill_H(self, R, n, E, H, iRate, alpha_exc):
        x = np.zeros((R.shape[0], 1))
        x[0] = math.sin(alpha_exc)
        x[1] = 0.0
        x[2] = math.cos(alpha_exc)
        for iEcho in range(n):
            x = np.dot(E, x)
            H[iEcho, iRate] = x[0]
        return H


    def sim_epg(self, nEchoes, TEmin, fa, T2grid_type='linear', t2_range=None):
        T1value = 1000.

        if T2grid_type == 'linear':
            if t2_range is not None:
                T2grid, dT2grid = np.linspace(t2_range[0], t2_range[1], 2000, retstep=True)
            else:
                T2grid, dT2grid = np.linspace(1., 2000., 2000, retstep=True)
            
        elif T2grid_type == 'log':
            if t2_range is not None:
                T2grid = np.geomspace(t2_range[0], t2_range[1], 60)
            else:
                T2grid = np.geomspace(10., 2000., 60)

        T1grid = T1value * np.ones_like(T2grid)

        R1 = np.array(1.0 / T1grid)
        R2 = np.array(1.0 / T2grid)
        rad = np.pi / 180.0

        epg = self.epg_signal(nEchoes, TEmin, R1, R2, fa * rad, 90. * rad)
        return epg


    def sim_multiple_epgs(self):
        
        epgs = []
        fa_list = self.fa_list
        for fa in fa_list:
            epg = self.sim_epg(self.nEchoes, self.TEmin, fa, self.T2grid_type, self.t2_range)
            epgs.append(epg)
        return np.array(epgs)


     
def get_pixelwise_models(model_type, n_echoes=10, out_ch=60, device='cuda'):
    if model_type == 'P2T2-FC':
        input_channel = n_echoes *2
        model = MLP_Model(
            input_channel=input_channel,
            channels=[256] * 12,
            output_channel=out_ch
        ).to(device)
    elif model_type == 'MIML':
        input_channel = n_echoes
        model = MLP_Model(
            input_channel=input_channel,
            channels=[256] * 6,
            output_channel=out_ch
        ).to(device)

    return model

