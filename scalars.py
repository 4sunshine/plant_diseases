import torch
import os
import numpy as np
from dataset import PlantDiseaseDataset


def dotter():
    dataset_info = '/home/sunshine/irishka/0.3v/json/src_ds.json'
    root_dir = '/home/sunshine/irishka/0.3v/ds/src'
    dataset = PlantDiseaseDataset(dataset_info, root_dir)
    state = torch.load(os.path.join(root_dir, 'train_glcm.pth'))
    features = torch.cat(state['features'], dim=0)
    features = [f.numpy() for f in features]
    labels = [l.item() for l in state['labels']]

    print(features[0].shape)

    label_features = {}
    for l, f in zip(labels, features):
        key = dataset.label_map(l)
        if key not in label_features.keys():
            label_features[key] = []
        label_features[key].append(f)

    healthy = 'Tomato___healthy'

    all_healthy = np.array(label_features.pop(healthy))

    means = [np.mean(all_healthy, axis=0).T]

    for k, v in label_features.items():
        plants = np.array(v)
        means += [np.mean(plants, axis=0).T]

    means = np.concatenate(means)
    means_norm = np.sqrt(np.sum(np.multiply(means, means), axis=1))
    means_norm = np.expand_dims(means_norm, axis=-1)
    means = means / means_norm
    return np.dot(means, means.T)

