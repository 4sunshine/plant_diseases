import numpy as np
import torch.nn as nn
import torch


def all_sliding_windows(a, stride=[4, 4], mask_size=[17, 17]):
    """RETURNS ARRAY OF SHAPE (B, C, Nx, Ny, MaskX, MaskY)"""
    shape = a.shape[:-2] + ((a.shape[-2] - mask_size[-2]) // stride[-2] + 1, ) + \
            ((a.shape[-1] - mask_size[-1]) // stride[-1] + 1,) + tuple(mask_size)
    strides = a.strides[:-2] + (a.strides[-2] * stride[-2],) + (a.strides[-1] * stride[-1],) +\
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def statistics(a, axis=(-2, -1)):
    means = a.mean(axis=axis)
    std = a.std(axis=axis)
    maximums = a.max(axis=axis)
    minimums = a.min(axis=axis)
    return means, std, maximums, minimums


def quantize_stat(a, mean, std, mx, mn):
    bins = np.array([0, mean-std, mean, mean+std, mx])






class Leafnet(nn.Module):
    """REQUIRES TENSOR INPUT"""
    def __init__(self):
        super().__init__()
        self.layer1 = GLCM()
        #self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = self.layer1(x)
        return x


class GLCM(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, mask_size=[17, 17], stride=[4, 4], images_size=[256, 256]):
        super().__init__()
        assert (mask_size[0] % 2 == 1) and (mask_size[1] % 2 == 1)
        mask_size = torch.Tensor(mask_size)
        self.mask_size = nn.Parameter(mask_size, requires_grad=False)
        stride = torch.Tensor(stride)
        self.stride = nn.Parameter(stride, requires_grad=False)
        images_size = torch.Tensor(images_size)
        self.images_size = nn.Parameter(images_size, requires_grad=False)
        #  side_pad = (self.images_size - 2 * (self.mask_size - 1) + 1) % self.stride
        #  self.side_pad = nn.Parameter(side_pad, requires_grad=False)

    def forward(self, images):
        images = images.detach().numpy()
        b, c, h, w = images.shape
        assert (h == self.images_size[0] + 1) and (w == self.images_size[0] + 1)
        sw = all_sliding_windows(images)
        m, s, mx, mn = statistics(sw)
        sw_q = quantize_stat(sw, statistics(sw))
        return np.shape(m)

