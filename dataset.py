import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def splitted_loaders(dataset, train_size, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_size * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=valid_sampler, shuffle=False)
    return train_loader, test_loader


# def shuffled_arrays(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p.astype(int)], b[p.astype(int)]


class PlantDiseaseDataset(Dataset):

    def __init__(self, json_file, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.paths, self.labels, self.map = self._load_info(json_file)

    def _load_info(self, json_file):
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

        return {'images': image, 'labels': self.labels[idx], 'path': img_name}

