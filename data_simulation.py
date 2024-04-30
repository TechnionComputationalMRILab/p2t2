import numpy as np
import math
import os
import pickle
import argparse
import yaml
from box import Box
from scipy.stats import norm

"""
Based on the data simulation presented in MIML: 
    Thomas Yu, Erick Jorge Canales-Rodríguez, Marco Pizzolato, Gian Franco Piredda, Tom Hilbert, Elda Fischi-Gomez, Matthias Weigel, Muhamed Barakovic, Meritxell Bach Cuadra, Cristina Granziera, Tobias Kober, Jean-Philippe Thiran,
    Model-informed machine learning for multi-component T2 relaxometry,
    Medical Image Analysis,
    Volume 69,
    2021,
    101940,
    ISSN 1361-8415,
    https://doi.org/10.1016/j.media.2020.101940.

    ------------------------------------------------------------------------------
    Robust myelin water imaging from multi-echo T2 data using second-order Tikhonov regularization with control points
    ISMRM 2019, Montreal, Canada. Abstract ID: 4686
    ------------------------------------------------------------------------------
    Developers:
    
    Erick Jorge Canales-Rodríguez (EPFL, CHUV, Lausanne, Switzerland; FIDMAG Research Foundation, CIBERSAM, Barcelona, Spain)
    Marco Pizzolato               (EPFL)
    Gian Franco Piredda           (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
    Tom Hilbert                   (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
    Tobias Kober                  (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
    Jean-Philippe Thiran          (EPFL, UNIL, CHUV, Switzerland)
    Alessandro Daducci            (Computer Science Department, University of Verona, Italy)
    Date: 11/02/2019
    ------------------------------------------------------------------------------
"""

class EPG_Generator:
    """
    
    """
    def __init__(self, n_echoes, TEmin, TEmax, T2grid, T1value):
        self.n_echoes = n_echoes
        self.TEmin = TEmin
        self.TEmax = TEmax
        self.T2grid = T2grid
        self.T1grid = T1value * np.ones_like(self.T2grid)
        self.rad = np.pi / 180.0

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

    def sim_multiple_epgs_for_miml(self,):

        R1 = np.array(1.0 / self.T1grid)
        R2 = np.array(1.0 / self.T2grid)

        epgs = []
        fa_list = np.arange(90, 180, 0.25)
        for fa in fa_list:
            epg = self.epg_signal(self.n_echoes, self.TEmin, R1, R2, fa * self.rad, 90. * self.rad)
            epgs.append(epg)
        return np.array(epgs), fa_list
    
    def sim_multiple_epgs_for_p2t2(self, num_signals=10000):
        R1 = np.array(1.0 / self.T1grid)
        R2 = np.array(1.0 / self.T2grid)
        epgs = []
        fa_list = np.arange(90, 180, 0.25)
        te_list = np.arange(self.TEmin, self.TEmax, 0.1)
        # sample num_signals from the te-fa pairs
        te_fa_pairs = [(te, fa) for te in te_list for fa in fa_list]
        te_fa_pairs_inx = np.random.choice(len(te_fa_pairs), num_signals)
        te_fa_pairs_sampled = [te_fa_pairs[inx] for inx in te_fa_pairs_inx]
        for te, fa in te_fa_pairs_sampled:
            epg = self.epg_signal(self.n_echoes, te, R1, R2, fa * self.rad, 90. * self.rad)
            epgs.append(epg)
        return np.array(epgs), te_fa_pairs_sampled

class SimualteData:

    def __init__(self, args, n_echoes, te_min, te_max, out_folder, model_type='P2T2', num_signals=10000) -> None:
        super().__init__()

        T2_range = args.T2_linear.range
        self.T2grid, self.dT2grid = np.linspace(T2_range[0], T2_range[1], args.T2_linear.num_elm, retstep=True)


        epg_generator = EPG_Generator(n_echoes, te_min, te_max, self.T2grid, args.T1value)
        if model_type == 'P2T2':
            epgs, vTE_FA = epg_generator.sim_multiple_epgs_for_p2t2(num_signals)
            vTE, vFA = zip(*vTE_FA)
        else:
            epgs, vFA =  epg_generator.sim_multiple_epgs_for_miml()
        # save the epgs, vFA and vTE to numpy files zipped
        np.savez_compressed(file=os.path.join(out_folder, 'epgs.npz'), epgs=epgs)
        np.savez_compressed(os.path.join(out_folder, 'vFA.npz'), vFA)
        if model_type == 'P2T2':
            np.savez_compressed(os.path.join(out_folder, 'vTE.npz'), vTE)

        self.epgs = epgs
        self.vFA = vFA
        self.vTE = vTE if model_type == 'P2T2' else None

        self.args = args
        self.n_echoes = n_echoes
        self.te_min = te_min
        self.out_folder = out_folder

        
        Npc = self.args.T2_log.num_samples
        T2s = np.logspace(math.log10(self.args.T2_log.start), math.log10(self.args.T2_log.end),
                        num=Npc, endpoint=True, base=10.0)
        
        self.Npc = Npc
        self.T2s = T2s

        
    def Signal_MET2_dist_based_precomputed_EPG(self, dist, Km, epg_signal):
        """
        Related to the multi-echo signal
        :param epg_signal: EPG [n_echoes, len(p(T2))]
        :param dist: p(T2) distribution
        :param Km: (Spin-density x global constant)
        :return: Ms = Km*EPG*p(T2)
        """
        Ms = Km * epg_signal * dist
        Ms = np.sum(Ms, axis=1)
        return Ms

    def add_Rician_nois(self, signal, SNR):
        """

        :param signal:
        :param SNR:
        :return: signal_noise: noisy signal
        """

        sigma_met2 = signal[0] / SNR
        noise_met2_1 = np.random.normal(0, sigma_met2, signal.shape)
        noise_met2_2 = np.random.normal(0, sigma_met2, signal.shape)
        signal_noise = np.sqrt((signal + noise_met2_1) ** 2 + noise_met2_2 ** 2)

        return signal_noise

    def transform_dist_to_low_res(self, dist, T2s, Npc, dT2grid, T2grid):
        """

        :param dist:    p(T2)
        :param T2s:     log-grid T2 values [10.,2000.]
        :param Npc:     T2_log - num_samples
        :param dT2grid: linear grid - delta(T2)
        :param T2grid:  linear grid T2 values

        :return: signal_noise: noisy signal
        """
        dist2 = np.zeros((Npc))
        deltaT2 = np.zeros((Npc))
        T2_delta_max = T2s[0] + (T2s[1] - T2s[0]) / 2.0
        ind_0 = T2grid < T2_delta_max
        dist2[0] = np.sum(dist[ind_0] * dT2grid)
        deltaT2[0] = T2_delta_max
        # --- ind from 1 to Npc-1
        for it in range(1, Npc - 1):
            T2_delta_min = T2s[it - 1] + (T2s[it] - T2s[it - 1]) / 2.0
            T2_delta_max = T2s[it] + (T2s[it + 1] - T2s[it]) / 2.0
            ind_i = (T2grid >= T2_delta_min) & (T2grid < T2_delta_max)
            dist2[it] = np.sum(dist[ind_i] * dT2grid)
            deltaT2[it] = T2_delta_max - T2_delta_min
        T2_delta_min = T2s[Npc - 2] + (T2s[Npc - 1] - T2s[Npc - 2]) / 2.0
        ind_Npc = T2grid >= T2_delta_min
        dist2[Npc - 1] = np.sum(dist[ind_Npc] * dT2grid)
        deltaT2[Npc - 1] = (T2s[Npc - 1] - T2s[Npc - 2]) / 2.0
        # normalization
        dist2 = np.convolve(dist2, np.ones(3) / 3,
                            mode='same')  # small smoothing to avoid oscilations due to discretization errors
        dist2 = dist2 / np.sum(dist2)
        return dist2

    def get_component_mu_and_std(self, data_args, N_voxels, component='Myelin'):
        args = data_args[component]
        mean_range = args.t2_mean
        std_range = args.t2_std
        mu = (np.random.uniform(low=mean_range[0], high=mean_range[1], size=N_voxels))
        std = (np.random.uniform(low=std_range[0], high=std_range[1], size=N_voxels))

        return mu, std

    def get_water_pool_params(self, data_args, water_pool_name='Myelin'):
        """

        :param data_args:
        :param water_pool_name: name of water pool case:
            WM = White Matter: myelin + ies
            GM = Gray Matter: myelin +  GM
            CSF = Cerebrospinal fluid
            WM_CSF: WM + CSF
            WM_GM: WM + GM
            CSF_GM: GN + CSF
            pathology
        :return: mu, std and volume fraction for each component in the water pool
        """
        N_voxels = data_args.num_signals  # 200,000
        water_pool_args = data_args.water_pool[water_pool_name]
        volume_fractions = np.random.dirichlet(alpha=water_pool_args.derichlet_alpha, size=N_voxels)

        if water_pool_name == 'GM':
            # change the vi of myelin to 0-5%
            volume_fractions[:, 0] *= (5 / 100)
            volume_fractions[:, 1] = 1 - volume_fractions[:, 0]
        if water_pool_name == 'CSF_GM':
            # change the vi of myelin to 0-5%
            volume_fractions[:, 0] *= (5 / 100)
            non_myelin_frac = 1 - volume_fractions[:, 0]
            volume_fractions[:, 1] *= non_myelin_frac
            volume_fractions[:, 2] = 1 - (volume_fractions[:, 0] + volume_fractions[:, 1])
        
        water_pool_volume_fraction = {key:None for key in data_args.components}
        water_pool_mu = {key:None for key in data_args.components}
        water_pool_std = {key:None for key in data_args.components}

        for i, component in enumerate(water_pool_args.components):
            v_component = volume_fractions[:, i]
            mu_component, std_component = self.get_component_mu_and_std(data_args, N_voxels, component=component)
            water_pool_volume_fraction[component] = v_component
            water_pool_mu[component] = mu_component
            water_pool_std[component] = std_component

        return water_pool_volume_fraction, water_pool_mu, water_pool_std

    def get_dist(self, water_pool_args, data_args, T2grid, Vf, Vmu, Vsigma, ind):
        v = {key:0.0 for key in data_args.components}
        mean = {key:0.0 for key in data_args.components}
        std = {key:0.0 for key in data_args.components}

        for i, component in enumerate(water_pool_args.components):
            frac = Vf[component][ind]
            mu = Vmu[component][ind]
            sigma = Vsigma[component][ind]

            v[component] = frac
            mean[component] = mu
            std[component] = sigma
            if i == 0:
                dist = frac * norm.pdf(T2grid, mu, sigma)
            else:
                dist = dist + frac * norm.pdf(T2grid, mu, sigma)

        dist = dist / np.sum(dist)
        return dist, v, mean, std

    def simulate_wp_data(self, water_pool_name):
        
        wp_out_folder = os.path.join(self.out_folder, water_pool_name)
        os.makedirs(wp_out_folder, exist_ok=True)

        water_pool_args = self.args.water_pool[water_pool_name]

        vKm = np.random.uniform(low=100.0, high=1000.0, size=self.args.num_signals)
        vSNR = np.random.uniform(low=self.args.low_SNR, high=self.args.high_SNR, size=self.args.num_signals)
        
        Vf, Vmu, Vsigma = self.get_water_pool_params(self.args, water_pool_name=water_pool_name)

        np.savez(os.path.join(wp_out_folder, f'{water_pool_name}_Vf.npz'), Vf)
        print('save Vf')
        np.savez(os.path.join(wp_out_folder, f'{water_pool_name}_Vmu.npz'), Vmu)
        print('save V mu')
        np.savez(os.path.join(wp_out_folder, f'{water_pool_name}_Vsigma.npz'), Vsigma)
        print('save V sigma')
        np.savez(os.path.join(wp_out_folder, f'{water_pool_name}_vKm.npz'), vKm)
        print('save V km')
        np.savez(os.path.join(wp_out_folder, f'{water_pool_name}_vSNR.npz'), vSNR)
        print('save V SNR')

        # --- get T2 distribution
        data = []
        Data = np.zeros((self.args.num_signals, self.n_echoes))
        Distributions = np.zeros((self.args.num_signals, self.Npc))
        print('Fitting...please wait')
        for iter in range(0, self.args.num_signals):
            # # --- get p(T2) dist
            dist, v, mean, std = self.get_dist(water_pool_args, self.args, self.T2grid, Vf, Vmu, Vsigma, iter)
            # # --- get MRI signal
            Km = vKm[iter]
            # if epg_path is not None:
            ind = np.random.randint(0, self.epgs.shape[0])
            signal_all = self.epgs[ind, :, :]
            FA = self.vFA[ind]
            if self.vTE is not None:
                TE_array = self.vTE[ind]* np.arange(1, self.n_echoes + 1)
            else:
                TE_array = self.te_min * np.arange(1, self.n_echoes + 1)
            
            signal = self.Signal_MET2_dist_based_precomputed_EPG(dist, Km, signal_all)

            # # --- add noise
            SNR = vSNR[iter]
            signal_noise = self.add_Rician_nois(signal, SNR)
            Data[iter, :] = signal_noise
            km_norm = signal_noise[0]
            s_norm = signal_noise / km_norm  # normalization (to obtain a signal with a proton density = 1)

            # # --- Transform the original high resolution pdf to the low resolution pdf
            dist2 = self.transform_dist_to_low_res(dist, self.T2s, self.Npc, self.dT2grid, self.T2grid)
            Distributions[iter, :] = dist2

            iter_data = {'s': signal_noise, 's_norm': s_norm, 'TE': TE_array, 'p_T2': dist2,
                        'alpha': FA, 'v': v, 'mean': mean, 'std': std, 'Km': Km, 'SNR': SNR}
            data.append(iter_data)

        np.savez(os.path.join(wp_out_folder, f'{water_pool_name}_distributions.npz'), Distributions)
        np.savez(os.path.join(wp_out_folder, f'{water_pool_name}_sample_signals.npz'), Data)
        out_name = f"{water_pool_name}_data.pkl"
        data_file = open(os.path.join(wp_out_folder, out_name), "wb")
        pickle.dump(data, data_file)
        data_file.close()
        print('done')
        return

    def simulate_data(self, ):
        water_pool_names = self.args.water_pool.keys()
        for water_pool_name in water_pool_names:
            print(water_pool_name)
            self.simulate_wp_data(water_pool_name)



def main(config_path, min_te, max_te, n_echoes, out_folder, model_type='P2T2', num_signals=10000):

    with open(config_path) as f:
        data_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    
    data_args.min_te = min_te
    data_args.max_te = max_te
    data_args.n_echoes = n_echoes
    data_args.model_type = model_type



    # save the data_args
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, 'data_args.yaml'), 'w') as f:
        yaml.dump(data_args.to_dict(), f)

    data_simulator = SimualteData(data_args, n_echoes, min_te, max_te, out_folder, model_type, num_signals)
    data_simulator.simulate_data()


if __name__ == '__main__':
    # config_path = 'mice_data_confige.yaml'
    # min_te = 11.0
    # n_echoes = 10
    # out_folder = 'mice_data_11_0ms_miml'
    # main(config_path, min_te, None, n_echoes, out_folder, model_type='MIML')

    # simulate multi te data
    # config_path = 'brain_data_config.yaml'


    # config_path = 'config.yaml'
    # min_te = 5.0
    # max_te = 15.0
    # n_echoes = 20
    # num_signals = 10000
    # out_folder = 'brain_data_5_15ms_p2t2'
    # main(config_path, min_te, max_te, n_echoes, out_folder, model_type='P2T2', num_signals=num_signals)

    parser = argparse.ArgumentParser(description='Data Simulation')
    parser.add_argument('--config', type=str, help='path to the config file', default='config.yaml')
    parser.add_argument('--min_te', type=float, help='minimum echo time', default=5.0)
    parser.add_argument('--max_te', type=float, help='maximum echo time (ignored for MIML)', default=15.0)
    parser.add_argument('--n_echoes', type=int, help='number of echoes (ignored for P2T2)', default=20)
    parser.add_argument('--model_type', type=str, help='model type', default='P2T2')
    parser.add_argument('--num_signals', type=int, help='number of signals', default=10000)
    parser.add_argument('--out_folder', type=str, help='output folder', default='data')

    args = parser.parse_args()

    print("config: ", args.config)
    print("model_type: ", args.model_type)
    print("min_te: ", args.min_te)

    if args.model_type == 'MIML':
        print("max_te: ", args.max_te)

    if args.model_type == 'P2T2':
        print("n_echoes: ", args.n_echoes)

    print("num_signals: ", args.num_signals)
    print("out_folder: ", args.out_folder)

    main(
        config_path=args.config,
        min_te=args.min_te,
        max_te=args.max_te,
        n_echoes=args.n_echoes,
        out_folder=args.out_folder,
        model_type=args.model_type,
        num_signals=args.num_signals
    )
