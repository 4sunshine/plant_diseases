import torch
from torchvision import transforms, utils
from dataset import PlantDiseaseDataset, splitted_loaders
import numpy as np
import random
from model import Leafnet
from utils import nonzero_stats, aggregate_stats
from tqdm import tqdm

if __name__ == '__main__':
    '''TESTING SETTINGS'''
    np.random.seed(42)
    random.seed(1001)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1002)

    '''DATASET PARAMS'''
    dataset_info = '/home/sunshine/irishka/0.3v/json/src_ds.json'
    root_dir = '/home/sunshine/irishka/0.3v/ds/src'
    batch_size = 32
    train_size = .75
    n_jobs = 8  # N JOBS FOR GLCM
    #  STATISTICS FOR QUANTIZATION  ##################################################
    #  CALCULATED PARAMETERS. MEAN NONZERO RED = 85.384; MEAN NONZERO RED STD = 53.798
    #  m, s = aggregate_stats(train_loader)
    #  ?? NOT IMPLEMENTED: PROPOSE NORMALIZATION WITH PARAMETERS ABOVE
    #  PROPOSE QUANTIZATION ON 5 GLOBAL BINS [0, .5, MEAN-STD, MEAN, MEAN+STD, 255]
    mean = 85.384
    std = 53.798
    bins = np.array([.0, .5, mean-std, mean, mean+std], dtype='float32')

    #  GLCM PROPERTIES
    glcm_d = [1, 3, 5]
    glcm_t = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm_l = np.shape(bins)[0]

    dataset = PlantDiseaseDataset(dataset_info, root_dir)

    '''LOADERS OF DATA'''
    train_loader, test_loader = splitted_loaders(dataset, batch_size=batch_size, train_size=train_size)

    '''MODEL'''
    model = Leafnet(bins=bins, global_mean=mean, global_std=std, n_jobs=n_jobs)

    with tqdm(total=len(train_loader)) as pbar:
        for i, data in enumerate(train_loader):
            output = model(data['images'])
            if i == 0:
                break
            pbar.update(1)

