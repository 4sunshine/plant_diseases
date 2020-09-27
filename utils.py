import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def nonzero_stats(images):
    """STATISTICS ON VALUES NOT EQUAL TO 0"""
    images = torch.flatten(images)
    n_nonzeros = images.numel() - (images == 0).sum()
    sum = images.sum()
    std = torch.std(images.type(torch.DoubleTensor))
    return n_nonzeros, sum, std


def aggregate_stats(data_loader):
    """COMMON STATS ACROSS ALL IMAGES"""
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
    std = torch.mean(torch.tensor(stds))

    return mean, std


def center_data(data, g_mean=None):
    if g_mean is None:
        mean = np.mean(data, axis=0)
        return mean, data - mean
    else:
        return data - g_mean


def whiten_data(data, g_std=None):
    if g_std is None:
        std = np.std(data, axis=0)
        return std, data / std
    else:
        return data / g_std


def train_and_test(ind, clf, train_data, test_data, center=True, whiten=True):
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    if ind:
        train_features = train_features[:, ind]
        test_features = test_features[:, ind]
    if center:
        train_mean, train_features = center_data(train_features)
        test_features = center_data(test_features, train_mean)
    if whiten:
        train_std, train_features = whiten_data(train_features)
        test_features = whiten_data(test_features, train_std)

    clf.fit(train_features, train_labels)
    prediction = clf.predict(test_features)

    pr, rec, fs, sup = precision_recall_fscore_support(test_labels, prediction, zero_division=0, average='binary')

    return clf, (pr, rec, fs)

