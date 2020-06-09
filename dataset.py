import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def features_labels(file):
    dataset = torch.load(file)
    features = dataset['features']
    labels = dataset['labels']
    #features = [f.numpy() for f in features]
    labels = [l.item() for l in labels]
    return features, labels


def splitted_loaders(dataset, train_size, batch_size):
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
        self.paths, self.labels, self.map = self._load_info(json_file)

    @staticmethod
    def _load_info(json_file):
        with open(json_file) as f:
            data = json.load(f)
        paths = [os.path.join(data['group_names'][g], f) for f, g in
                 zip(data['image_names'], data['group_id'])]
        return paths, data['group_id'], {k: v for k, v in enumerate(data['group_names'])}

    def label_map(self, label):
        return self.map[label]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.paths[idx])
        image = Image.open(img_name).convert('RGB')
        # H, W, [R,G,B]
        # RETURN ONLY RED CHANNEL
        image = np.array(image)[:, :, 0]
        image = np.expand_dims(image, axis=-1)
        # CONVERT TO PYTORCH CONVENTION:
        # N, C, H, W
        image = np.transpose(image, (2, 0, 1))
        # PAD IMAGES
        image = np.pad(image, [(0, 0), (0, 1), (0, 1)], mode='constant')
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return {'images': image, 'labels': self.labels[idx], 'paths': img_name}


class RawFeaturesDataset(Dataset):
    def __init__(self, data_file):
        # data file must contain state dict: {'features': [], 'labels': [], 'paths': []}
        data = torch.load(data_file)
        self.paths = data['paths']
        self.labels = data['labels']
        self.features = torch.cat(data['features'], dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {'features': self.features[i], 'labels': self.labels[i], 'paths': self.paths[i]}


