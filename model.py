import numpy as np
import torch.nn as nn
import torch


class Leafnet(nn.Module):
    '''REQUIRES NUMPY INPUT'''

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
        side_pad = (self.images_size - 2 * (self.mask_size - 1) + 1) % self.stride
        self.side_pad = nn.Parameter(side_pad, requires_grad=False)

    def forward(self, images):
        images = images.detach()
        b, c, h, w = images.shape
        assert (h == self.images_size[0]) and (w == self.images_size[0])

        return b

