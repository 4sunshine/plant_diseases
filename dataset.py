import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import cv2


def features_labels(file):
    dataset = torch.load(file)
    features = dataset['features']
    labels = dataset['labels']
    labels = [0 if (l.item() == 5) else 1 for l in labels]
    return features, labels


def splitted_loaders(dataset, train_size, batch_size):
    # TODO: BETTER SOLUTION THAN THIS IS TO: FIX TRAIN/TEST SPLITS IN EXTERNAL FILE
    # e.g. train_ds.json, test_ds.json and create two different datasets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_size * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=test_sampler, shuffle=False)
    return train_loader, test_loader


class PlantDiseaseDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.paths, self.labels = self._load_info(json_file)

    @staticmethod
    def _load_info(json_file):
        with open(json_file) as f:
            data = json.load(f)
        paths = [os.path.join(data['group_names'][g], f) for f, g in
                 zip(data['image_names'], data['group_id'])]
        return paths, data['group_id']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = os.path.join(self.root, self.paths[idx])
        image = cv2.imread(image_path)
        # H, W, [BGR]
        # RETURN ONLY RED CHANNEL
        image = image[:, :, -1]
        # CONVERT TO PYTORCH CONVENTION:
        # C, H, W
        image = np.expand_dims(image, axis=0)
        # PAD IMAGE
        image = np.pad(image, [(0, 0), (0, 1), (0, 1)], mode='constant')
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], image_path

