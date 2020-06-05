import torch
from tqdm import tqdm


def nonzero_stats(images):
    images = torch.flatten(images)
    n_nonzeros = images.numel() - (images == 0).sum()
    sum = images.sum()
    std = torch.std(images.type(torch.DoubleTensor))
    return n_nonzeros, sum, std


def aggregate_stats(data_loader):
    """COMMON STATS"""
    s = torch.tensor(0)
    count = torch.tensor(0)
    stds = []

    with tqdm(total=len(data_loader)) as pbar:
        for i, data in enumerate(data_loader):
            cnt, sm, std = nonzero_stats(data['images'])
            count += cnt
            s += sm
            stds += [std]
            pbar.update(1)
    mean = s / (count + 1e-7)
    #std = torch.cat(stds, dim=0)
    std = torch.mean(torch.tensor(stds))
    print(std)
    return mean, std

