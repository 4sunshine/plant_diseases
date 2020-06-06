import numpy as np
import torch.nn as nn
import torch
from skimage.feature import greycomatrix, greycoprops
from joblib import delayed, Parallel
import torch.nn as nn


def all_sliding_windows(a, stride=[4, 4], mask_size=[17, 17]):
    """RETURNS ARRAY OF SHAPE (B, C, Nx, Ny, MaskX, MaskY)"""
    shape = a.shape[:-2] + ((a.shape[-2] - mask_size[-2]) // stride[-2] + 1, ) + \
            ((a.shape[-1] - mask_size[-1]) // stride[-1] + 1,) + tuple(mask_size)
    strides = a.strides[:-2] + (a.strides[-2] * stride[-2],) + (a.strides[-1] * stride[-1],) +\
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def statistics(a, g_mean, g_std, axis=(-2, -1)):
    means = a.mean(axis=axis)
    std = a.std(axis=axis)
    maximums = a.max(axis=axis)
    minimums = a.min(axis=axis)
    means_ = torch.from_numpy((maximums - g_mean) / g_std)
    std = torch.from_numpy(std / g_std)
    maximums = torch.from_numpy((maximums - means) / g_std)
    minimums = torch.from_numpy((means - minimums) / g_std)
    # TENSOR OF SHAPE: B, 4, H, W
    # DIM 1: [MEAN, STD, MAX, MIN]
    return torch.cat([means_, std, maximums, minimums], dim=1)


#  NOT USED
def quantize_axis(a):
    mean = np.mean(a)
    print(mean)
    std = np.std(a)
    print(std)
    minima = np.min(a)
    maxima = np.max(a)
    # WEAK PLACE!!!
    # THINK ABOUT BIN
    bins = [minima, max(mean-std, minima), mean, mean+std, maxima]
    # CHECK IF ALL ELEMENTS SAME
    if len(set(bins)) == 1:
        bins += [minima + 1]
    # CHECK DIMS
    print(bins)
    return np.digitize(a, bins)


#  NOT USED
def quantize_stat(a):
    a = a.reshape(np.shape(a)[:-2] + (-1,))
    a = np.apply_along_axis(quantize_axis, -1, a)
    return np.shape(a)


class FeatureOperator(nn.Module):
    """REQUIRES TENSOR INPUT"""
    def __init__(self, kernel_size):
        super().__init__()
        # LAYERS
        self.a_pool = nn.AvgPool2d(kernel_size)
        self.m_pool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        a = self.a_pool(x)
        m = self.m_pool(x)
        x = torch.cat([a, m], dim=1)
        x = torch.squeeze(x)
        return x


class Leafnet(nn.Module):
    """REQUIRES TENSOR INPUT"""
    def __init__(self, bins, global_mean, global_std, dist, theta, levels, n_jobs):
        super().__init__()

        # LAYERS
        self.sw_layer = SlidingWindows()
        self.stat_layer = StatLayer(g_mean=global_mean, g_std=global_std)
        self.q_layer = Quantizer(bins=bins)
        self.glcm = GLCM(n_jobs=n_jobs, dist=dist, theta=theta, levels=levels)

    def forward(self, x):
        # x = TENSOR OF SHAPE: B, 1, H, W, MaskY, MaskX

        x = self.sw_layer(x)

        # s = TENSOR OF SHAPE: B, 4, H, W
        # DIM 1: [MEAN, STD, MAX, MIN]

        s = self.stat_layer(x)

        # q = QUANTIZED TENSOR OF SHAPE: B, 1, H, W, MaskY, MaskX

        q = self.q_layer(x)

        # g = GLCM PROPERTIES OF SHAPE: B, Nprop, H, W
        # DIM 1: [HIST, *GLCM PARAMS]

        g = self.glcm(q)

        # x = TENSOR OF SHAPE B, C, H, W
        # DIM 1: [STAT PROPS, HIST, GLCM PARAMS]

        x = torch.cat([s, g], dim=1)

        return x


class GLCM(nn.Module):
    """ GLCM PARALLEL! CONTAINS HIST CALCULATION """
    def __init__(self, dist, theta, levels, n_jobs=8):
        super().__init__()
        self.n_jobs = n_jobs
        self.dist = dist
        self.theta = theta
        self.levels = levels

    def glcm_prop(self, q):
        g = greycomatrix(q, self.dist, self.theta, self.levels, normed=True, symmetric=True)
        properties = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')
        props = np.array([greycoprops(g, p) for p in properties]).reshape(-1)
        # CALCULATE HIST HERE
        if np.max(q) > 0:
            hist, _ = np.histogram(q.reshape(-1), bins=np.arange(1, self.levels+1), density=True)
        else:
            hist = np.zeros(self.levels - 1)
        props = props.reshape(-1)
        return np.concatenate([hist, props], axis=0)

    def forward(self, q):
        sh = np.shape(q)
        gc = q.reshape((-1,) + sh[-2:])
        gcs = Parallel(n_jobs=self.n_jobs)(delayed(self.glcm_prop)(img) for img in gc)
        gcs = np.stack(gcs, axis=0)
        gcs = gcs.reshape(sh[:-2] + (-1,))
        gcs = gcs.transpose(0, -1, 2, 3, 1)
        gcs = gcs.squeeze(-1)
        return torch.from_numpy(gcs)


class SlidingWindows(nn.Module):
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
        _, _, h, w = images.shape
        assert (h == self.images_size[0] + 1) and (w == self.images_size[0] + 1)
        sw = all_sliding_windows(images)
        return sw


class StatLayer(nn.Module):
    def __init__(self, g_mean, g_std):
        super().__init__()
        self.g_mean = g_mean
        self.g_std = g_std

    def forward(self, sw):
        s = statistics(sw, g_mean=self.g_mean, g_std=self.g_std)
        return s


class Quantizer(nn.Module):
    def __init__(self, bins):
        super().__init__()
        bins = torch.from_numpy(bins)
        self.bins = nn.Parameter(bins, requires_grad=False)

    def forward(self, x):
        sh = np.shape(x)
        x = x.reshape(sh[:-2] + (-1,))
        x = np.digitize(x, self.bins.numpy()) - 1
        return x.reshape(sh)


