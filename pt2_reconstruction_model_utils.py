import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset


class wasserstein_distance(nn.Module):
    def __init__(self, min_T2=10., max_T2= 2000., num_samples=60):
        super(wasserstein_distance, self).__init__()

        """
        Implementation of the Wasserstein Distance
        :param batch_size:
        :param num_samples:
        :return:
        """
        self.arr = np.logspace(math.log10(min_T2), math.log10(max_T2), num=num_samples, endpoint=True, base=10.0)

    def forward(self, y_actual, y_pred):
        batch_size = y_actual.shape[0]
        arr = np.tile(self.arr, (batch_size, 1))
        arr_tensor = torch.FloatTensor(arr.astype('float32'))

        abs_cdf_difference = torch.abs(torch.cumsum(y_actual - y_pred, dim=1))  # tf.math.abs(tf.math.cumsum(y_actual -
        # arr_tensor = arr_tensor.to(y_pred.get_device())
        arr_tensor = arr_tensor.to(y_pred.device)
        wass_loss = torch.mean(
            0.5 * torch.sum(
                torch.multiply(
                    -arr_tensor[:, :-1] + arr_tensor[:, 1:],
                    abs_cdf_difference[:, :-1] + abs_cdf_difference[:, 1:]),
                dim=1)
        )
        return wass_loss


class MSE_wasserstein_combo(nn.Module):

    def __init__(self, min_T2=10., max_T2= 2000., num_samples=60):
        """
        Combination loss function used in MIML
        """
        super(MSE_wasserstein_combo, self).__init__()

        self.wass_loss_f = wasserstein_distance(min_T2=min_T2, max_T2=max_T2, num_samples=num_samples)
        self.mse_f = torch.nn.MSELoss()

    def forward(self, y_actual, y_pred, mse_weight=100000.0):
        wass_loss = self.wass_loss_f(y_actual, y_pred)
        mse = self.mse_f(y_pred, y_actual)
        return wass_loss + mse_weight * mse


class T2_Dataset(Dataset):

    def __init__(self, T2_data):
        self.T2_data = T2_data

    def __len__(self):
        return len(self.T2_data)

    def __getitem__(self, item):
        data = self.T2_data[item]
        return data
