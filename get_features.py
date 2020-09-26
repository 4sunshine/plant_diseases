import torch
from torchvision import transforms, utils
from dataset import PlantDiseaseDataset, splitted_loaders
import numpy as np
import random
from model import Leafnet, GLCMeR
from tqdm import tqdm
import os
from PIL import Image


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
    ds_mean = 85.384
    ds_std = 53.798
    # QUANTIZATION ON 5 GLOBAL BINS [0, .5, MEAN-STD, MEAN, MEAN+STD, 255]
    bins = np.array([.0, .5, ds_mean - ds_std, ds_mean, ds_mean + ds_std], dtype='float32')

    #  GLCM PROPERTIES
    glcm_d = [1, 2, 4]
    glcm_t = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm_l = np.shape(bins)[0]

    dataset = PlantDiseaseDataset(dataset_info, root_dir)

    '''LOADERS OF DATA'''
    train_loader, test_loader = splitted_loaders(dataset, batch_size=batch_size, train_size=train_size)

    '''MODEL'''
    model = Leafnet(bins=bins, global_mean=ds_mean, global_std=ds_std, n_jobs=n_jobs, dist=glcm_d,
                    theta=glcm_t, levels=glcm_l)

    # model = GLCMeR(dist=glcm_d, theta=glcm_t, levels=glcm_l, bins=bins)

    state = {'features': [], 'labels': [], 'paths': []}

    i = 0
    for data in tqdm(train_loader):
        train_file = os.path.join(root_dir, 'train_global_pooled.pth')
        a = torch.load(train_file)
        minima = torch.min(a['features'], dim=0)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        maxima = torch.max(a['features'], dim=0)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        output = model(data['images'])
        output = output.permute(1,0,2,3)
        output = 255*(output - minima)/(maxima-minima)
        output = utils.make_grid(output, nrow=8)
        output = output.permute(1, 2, 0).numpy()
        output = np.clip(output, a_min=0, a_max=255)
        output = np.uint8(np.round(output))
        #output = output.astype(int)
        print(np.shape(output))
        x = Image.fromarray(output)
        x = x.convert('LA')
        x_name = os.path.basename(data['paths'][0])
        x_name = x_name.replace('jpg','png')
        x_name = dataset.label_map(data['labels'][0].item()) + x_name
        x_name = os.path.join(root_dir, 'filters/', x_name)
        x.save(x_name)
        i += 1

        # state['features'] += [model(data['images'])]
        # state['labels'] += data['labels']
        # state['paths'] += data['paths']
        if i==0:
            break
    #
    # torch.save(state, os.path.join(root_dir, 'train_global_features.pth'))
    # print('Train Features Saved!')
    #
    # state = {'features': [], 'labels': [], 'paths': []}
    #
    # for data in tqdm(test_loader):
    #     #output = model(data['images'])
    #     state['features'] += [model(data['images'])]
    #     state['labels'] += data['labels']
    #     state['paths'] += data['paths']
    #
    # torch.save(state, os.path.join(root_dir, 'test_global_features.pth'))
    # print('Test Features Saved!')
    #
