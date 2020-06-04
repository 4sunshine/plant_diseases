import torch
from torchvision import transforms, utils
from utils import PlantDiseaseDataset, splitted_loaders


if __name__ == '__main__':
    '''DATASET PARAMS'''
    dataset_info = '/home/sunshine/irishka/0.3v/json/ds.json'
    root_dir = '/home/sunshine/irishka/0.3v/ds/src'
    batch_size = 16
    train_size = .75

    dataset = PlantDiseaseDataset(dataset_info, root_dir)

    '''LOADERS OF DATA'''
    train_loader, test_loader = splitted_loaders(dataset, batch_size=batch_size, train_size=train_size)


