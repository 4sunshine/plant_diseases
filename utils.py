import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class PlantDiseaseDataset(Dataset):

    def __init__(self, json_file, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.paths, self.labels = self._load_info(json_file)

    def _load_info(self, json_file):
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
        img_name = os.path.join(self.root, self.paths[idx])
        image = Image.open(img_name)

        if self.transform:
            sample = self.transform(image)

        return sample

