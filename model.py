import numpy as np
from skimage.feature import greycomatrix, greycoprops
from joblib import delayed, Parallel
from torch.nn import Module, Parameter
import torch


class Leafchik(Module):
    PROPERTIES = ('contrast', 'homogeneity', 'energy', 'correlation')
    STAT_AXIS = (-2, -1)

    def __init__(self, n_jobs=8, stride=[4, 4], mask_size=[17, 17], g_mean=85.384, g_std=53.798, dist=[1, 2, 4],
                 theta=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
        super().__init__()

        self.n_jobs = n_jobs

        self.stride = Parameter(torch.tensor(stride), requires_grad=False)
        self.mask_size = Parameter(torch.tensor(mask_size), requires_grad=False)
        self.g_mean = Parameter(torch.tensor(g_mean), requires_grad=False)
        self.g_std = Parameter(torch.tensor(g_std), requires_grad=False)
        self.bins = Parameter(torch.tensor([.0, .5, g_mean - g_std, g_mean, g_mean + g_std]), requires_grad=False)
        self.dist = Parameter(torch.tensor(dist), requires_grad=False)
        self.theta = Parameter(torch.tensor(theta), requires_grad=False)
        self.levels = Parameter(torch.tensor(len(self.bins)), requires_grad=False)
        self.params_to_numpy()

    # def state_dict(self):
    #     state = {
    #         'stride': self.stride,
    #         'mask_size': self.mask_size,
    #         'g_mean': self.g_mean,
    #         'g_std': self.g_std,
    #         'bins': self.bins,
    #         'dist': self.dist,
    #         'theta': self.theta,
    #         'levels': self.levels
    #     }
    #     return OrderedDict(state)

    def params_to_numpy(self):
        for name, param in self.named_parameters():
            # param = param.numpy()
            print(param)

    def all_sliding_windows(self, a):
        """RETURNS ARRAY OF SHAPE (B, C, Nx, Ny, MaskX, MaskY)"""
        shape = a.shape[:-2] + ((a.shape[-2] - self.mask_size[-2]) // self.stride[-2] + 1,) + \
                ((a.shape[-1] - self.mask_size[-1]) // self.stride[-1] + 1,) + tuple(self.mask_size)
        strides = a.strides[:-2] + (a.strides[-2] * self.stride[-2],) + (a.strides[-1] * self.stride[-1],) + \
                  a.strides[-2:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def statistics(self, a):
        g_mean = self.g_mean.numpy()
        g_std = self.g_std.numpy()
        means = a.mean(axis=self.STAT_AXIS)
        std = a.std(axis=self.STAT_AXIS)
        maximums = a.max(axis=self.STAT_AXIS)
        minimums = a.min(axis=self.STAT_AXIS)
        means /= g_mean
        std /= g_std
        maximums = (maximums - means) / g_std
        minimums = (means - minimums) / g_std
        return np.concatenate([means, std, maximums, minimums], axis=1)

    def quantize(self, x):
        sh = np.shape(x)
        x = x.reshape(sh[:-2] + (-1,))
        x = np.digitize(x, self.bins) - 1
        return x.reshape(sh)

    def _single_hist_glcm_calculate(self, q):
        levels = self.levels.numpy()
        g = greycomatrix(q, self.dist, self.theta, levels, normed=True, symmetric=True)
        props = np.array([greycoprops(g, p) for p in self.PROPERTIES]).reshape(-1)
        entropy = -np.sum(np.multiply(g, np.log2(g+1e-8)), axis=(0, 1)).reshape(-1)
        props = np.concatenate([props, entropy], axis=0)
        # CALCULATE HIST HERE
        if np.max(q) > 0:
            hist, _ = np.histogram(q.reshape(-1), bins=np.arange(1, levels + 1), density=True)
        else:
            hist = np.zeros(levels - 1)
        return np.concatenate([hist, props], axis=0)

    def hist_glcm(self, q):
        sh = np.shape(q)
        gc = q.reshape((-1,) + sh[-2:])
        gcs = Parallel(n_jobs=self.n_jobs)(delayed(self._single_hist_glcm_calculate)(img) for img in gc)
        gcs = np.stack(gcs, axis=0)
        gcs = gcs.reshape(sh[:-2] + (-1,))
        gcs = gcs.transpose(0, -1, 2, 3, 1)
        return gcs.squeeze(-1)

    def forward(self, inputs):
        x = inputs.numpy()

        # x = NUMPY_NDARRAY OF SHAPE: B, 1, H, W, MaskY, MaskX
        x = self.all_sliding_windows(x)

        # stat = NUMPY_NDARRAY OF SHAPE: B, [MEAN, STD, MAX, MIN], H, W
        stat = self.statistics(x)

        # quant = NUMPY_NDARRAY OF SHAPE: B, 1, H, W, MaskY, MaskX
        quant = self.quantize(x)

        # hist_glcm = NUMPY_NDARRAY OF SHAPE: B, [HIST, GLCM], H, W
        hist_glcm = self.hist_glcm(quant)

        # features = NUMPY_NDARRAY OF SHAPE: B, [STAT, HIST, GLCM], H, W
        features = np.concatenate([stat, hist_glcm], axis=1)

        # numpy_average_pooling
        return np.mean(features, axis=(-2, -1), keepdims=False)

