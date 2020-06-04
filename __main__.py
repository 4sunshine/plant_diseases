import torch
from torchvision import transforms, utils
from utils import PlantDiseaseDataset, splitted_loaders
import numpy as np
import random


if __name__ == '__main__':
    '''TESTING SETTINGS'''
    np.random.seed(42)
    random.seed(1000)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1001)

    '''DATASET PARAMS'''
    dataset_info = '/home/sunshine/irishka/0.3v/json/src_ds.json'
    root_dir = '/home/sunshine/irishka/0.3v/ds/src'
    batch_size = 16
    train_size = .75

    dataset = PlantDiseaseDataset(dataset_info, root_dir)

    '''LOADERS OF DATA'''
    train_loader, test_loader = splitted_loaders(dataset, batch_size=batch_size, train_size=train_size)

    for i, data in enumerate(train_loader):
        print(np.shape(data['images']))
        if i == 3:
            break

