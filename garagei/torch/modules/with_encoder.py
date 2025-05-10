import numpy as np

import torch
from torch import nn

from garagei.torch.modules.spectral_norm import spectral_norm


class NormLayer(nn.Module):
    def __init__(self, name, dim=None):
        super().__init__()
        if name == 'none':
            self._layer = None
        elif name == 'layer':
            assert dim != None
            self._layer = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if self._layer is None:
            return features
        return self._layer(features)


class CNN(nn.Module):
    def __init__(self, num_inputs, act=nn.ELU, norm='none', cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=(400, 400, 400, 400), spectral_normalization=False):
        super().__init__()

        self._num_inputs = num_inputs
        self._act = act()
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

        self._conv_model = []
        for i, kernel in enumerate(self._cnn_kernels):
            if i == 0:
                prev_depth = num_inputs
            else:
                prev_depth = 2 ** (i - 1) * self._cnn_depth
            depth = 2 ** i * self._cnn_depth
            if spectral_normalization:
                self._conv_model.append(spectral_norm(nn.Conv2d(prev_depth, depth, kernel, stride=2)))
            else:
                self._conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
            self._conv_model.append(NormLayer(norm, depth))
            self._conv_model.append(self._act)
        self._conv_model = nn.Sequential(*self._conv_model)

    def forward(self, data):
        output = self._conv_model(data)
        output = output.reshape(output.shape[0], -1)
        return output


class Encoder(nn.Module):
    def __init__(
            self,
            pixel_shape,
            spectral_normalization=False,
            cnn_kernels=(4,4,4,4)
    ):
        super().__init__()

        self.pixel_shape = pixel_shape
        self.pixel_dim = np.prod(pixel_shape)

        self.pixel_depth = self.pixel_shape[-1]

        self.encoder = CNN(self.pixel_depth, spectral_normalization=spectral_normalization, cnn_kernels=cnn_kernels)

    def forward(self, input):
        unsqueezed = len(input.shape) == 1
        if unsqueezed:
            input = input.unsqueeze(0)

        assert len(input.shape) == 2

        pixel = input[..., :self.pixel_dim].reshape(-1, *self.pixel_shape).permute(0, 3, 1, 2)
        state = input[..., self.pixel_dim:]

        pixel = pixel / 255.

        rep = self.encoder(pixel)
        rep = rep.reshape(rep.shape[0], -1)
        output = torch.cat([rep, state], dim=-1)

        if unsqueezed:
            output = output.squeeze(0)

        return output


class CNNWithSkipConn(nn.Module):
    def __init__(self, num_inputs, norm='none', channels_list=(32, 64, 128, 256), kernels_list=(4, 4, 4, 4),
                 spectral_normalization=False):
        super().__init__()

        self._num_inputs = num_inputs
        self._norm = norm
        self._cnn_kernels = kernels_list

        self._layers = nn.ModuleList()
        self._norms = nn.ModuleList()
        self._projections = nn.ModuleList()

        prev_channels = num_inputs
        for channels, kernel in zip(channels_list, self._cnn_kernels):
            # Main convolutional layer
            if spectral_normalization:
                self._layers.append(spectral_norm(nn.Conv2d(prev_channels, channels, kernel, stride=2)))
            else:
                self._layers.append(nn.Conv2d(prev_channels, channels, kernel, stride=2))

            # Normalization layer
            self._norms.append(NormLayer(norm, channels))

            # Projection for skip connection
            self._projections.append(nn.Conv2d(prev_channels, channels, kernel, stride=2, bias=False))

            prev_channels = channels

    def forward(self, data):
        x = data

        for i in range(len(self._layers)):
            identity = x

            # Main path: Conv -> Norm -> LeakyReLU
            x = self._layers[i](x)
            x = self._norms[i](x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.25)

            # Skip connection with projection
            x = x + self._projections[i](identity)

        return x.reshape(x.shape[0], -1)


class EncoderWithSkipConn(nn.Module):
    def __init__(
            self,
            pixel_shape,
            spectral_normalization=False,
            channels_list=(81,81),
            kernels_list=(4,4),
    ):
        super().__init__()

        self.pixel_shape = pixel_shape
        self.pixel_dim = np.prod(pixel_shape)
        self.pixel_depth = self.pixel_shape[-1]

        self.encoder = CNNWithSkipConn(
            self.pixel_depth,
            spectral_normalization=spectral_normalization,
            kernels_list=kernels_list,
            channels_list=channels_list
        )

    def forward(self, input):
        unsqueezed = len(input.shape) == 1
        if unsqueezed:
            input = input.unsqueeze(0)

        assert len(input.shape) == 2

        pixel = input[..., :self.pixel_dim].reshape(-1, *self.pixel_shape).permute(0, 3, 1, 2)
        state = input[..., self.pixel_dim:]

        pixel = pixel / 255.

        rep = self.encoder(pixel)
        rep = rep.reshape(rep.shape[0], -1)
        output = torch.cat([rep, state], dim=-1)

        if unsqueezed:
            output = output.squeeze(0)

        return output


class WithEncoder(nn.Module):
    def __init__(
            self,
            encoder,
            module,
    ):
        super().__init__()

        self.encoder = encoder
        self.module = module

    def get_rep(self, input):
        return self.encoder(input)

    def forward(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module(rep, *inputs[1:])

    def forward_mode(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module.forward_mode(rep, *inputs[1:])
