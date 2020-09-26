import torch
from dataset import PlantDiseaseDataset, splitted_loaders
import numpy as np
import random
from model import Leafchik
from tqdm import tqdm
import os


if __name__ == '__main__':
    '''TESTING SETTINGS'''
    np.random.seed(42)
    random.seed(1001)
    torch.manual_seed(1002)

    '''DATASET PARAMS'''
    # TODO: USE HERE ARGPARSE LIB
    dataset_info = '/home/sunshine/irishka/0.3v/json/src_ds.json'
    root_dir = '/home/sunshine/irishka/0.3v/ds/src'
    batch_size = 20
    train_size = 0.8
    n_jobs = 8  # N JOBS FOR GLCM
    # END OF ARGPARSE PARAMS

    # STATISTICS FOR QUANTIZATION  ##################################################
    # CALCULATED PARAMETERS. MEAN NONZERO RED = 85.384; MEAN NONZERO RED STD = 53.798
    # HOW WAS ABOVE PARAMETERS OBTAINED:
    # ds_mean, ds_std = aggregate_stats(train_loader)  FROM UTILS.PY
    # QUANTIZATION ON 5 GLOBAL BINS [0, .5, MEAN-STD, MEAN, MEAN+STD, 255]
    # SEE ALL INFORMATION IN DEFAULT MODEL CONSTRUCTOR

    dataset = PlantDiseaseDataset(dataset_info, root_dir)

    '''LOADERS OF DATA'''
    train_loader, test_loader = splitted_loaders(dataset, batch_size=batch_size, train_size=train_size)

    '''MODEL'''
    model = Leafchik()

    data_to_save = {'features': [], 'labels': [], 'paths': []}

    for images, labels, images_paths in tqdm(train_loader, desc='Getting train features'):
        features = model(images)
        data_to_save['features'] += [features]
        data_to_save['labels'] += labels
        data_to_save['paths'] += images_paths

    data_to_save['features'] = np.concatenate(data_to_save['features'], axis=0)

    torch.save(data_to_save, os.path.join(root_dir, 'new_train_features.pth'))

    print('Train features successfully saved')
