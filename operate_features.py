import torch
import numpy as np
import os
from dataset import RawFeaturesDataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from model import FeatureOperator


'''TESTING SETTINGS'''
np.random.seed(42)
random.seed(1001)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1002)

root_dir = '/home/sunshine/irishka/0.3v/ds/src'
train_data = os.path.join(root_dir, 'train_features.pth')
test_data = os.path.join(root_dir, 'test_features.pth')

train_dataset = RawFeaturesDataset(train_data)
test_dataset = RawFeaturesDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

state = {'features': [], 'labels': [], 'paths': []}

with torch.no_grad():
    model = FeatureOperator(61)
    model.eval()

    with tqdm(total=len(train_loader)) as pbar:
        for i, data in enumerate(train_loader):
            # output = model(data['images'])
            state['features'] += [model(data['features'])]
            state['labels'] += data['labels']
            state['paths'] += data['paths']
            pbar.update(1)

        state['features'] = torch.cat(state['features'], dim=0)
        torch.save(state, os.path.join(root_dir, 'train_convpooled.pth'))
        print('Train Features Saved!')

    state = {'features': [], 'labels': [], 'paths': []}

    with tqdm(total=len(test_loader)) as pbar:
        for i, data in enumerate(test_loader):
            # output = model(data['images'])
            state['features'] += [model(data['features'])]
            state['labels'] += data['labels']
            state['paths'] += data['paths']
            pbar.update(1)

        state['features'] = torch.cat(state['features'], dim=0)
        torch.save(state, os.path.join(root_dir, 'test_convpooled.pth'))
        print('Test Features Saved!')

