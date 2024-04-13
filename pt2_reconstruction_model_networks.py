import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Model(nn.Module):

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


class Convolution_FC_Model(nn.Module):

    def __init__(self, input_shape, channels, output_channel):
        """
        :param input_shape: (2,32) = (2,n_echoes)
        :param channels: [256]*6
        :param output_channel: 120 = args.T2_log.num_samples (original = 60)
        """
        super().__init__()
        self.in_channel = input_shape
        self.channels = channels
        self.out_channel = output_channel

        self.net = self._create_net()

    def _create_net(self):
        input_dim = self.in_channel[0]  # 2
        input_channel = self.in_channel[-1]  # num_echoes
        channels = self.channels
        output_channel = self.out_channel

        n_layers = len(channels)
        layers = []
        # add convolution layer to project the input from 2D into 1D (2,n_echoes) -> (1,n_echoes)
        self.conv_1 = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=1, stride=1)
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
        x = self.conv_1(x)
        x = x.squeeze()
        output = self.net(x)
        return output
