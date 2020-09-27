import cv2
import torch
import numpy as np

from model import Leafchik, HealthyPlant


def load_and_preprocess(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    # H, W, [BGR]
    # RETURN ONLY RED CHANNEL
    image = image[:, :, -1]
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    # CONVERT TO PYTORCH CONVENTION:
    # C, H, W
    image = np.expand_dims(image, axis=0)
    # PAD IMAGE
    image = np.pad(image, [(0, 0), (0, 1), (0, 1)], mode='constant')
    return torch.from_numpy(image)


def load_model(backbone, backbone_path, sklearn_classifier_path):
    sklearn_classifier = torch.load(sklearn_classifier_path)
    classifier = sklearn_classifier['classifier']
    c_mean = sklearn_classifier['mean']
    c_std = sklearn_classifier['std']
    c_indices = sklearn_classifier['indices']
    backbone.load_state_dict(torch.load(backbone_path))
    return HealthyPlant(backbone, classifier, c_mean, c_std, c_indices)


if __name__ == '__main__':
    # TODO: ARGPARSE or SYS.ARGV
    backbone_path = 'PATH TO FEATURE EXTRACTOR'
    classifier_path = 'PATH TO SKLEARN CLASSIFIER'
    image_path = 'TEST IMAGE PATH'
    backbone = Leafchik()
    model = load_model(backbone, backbone_path, classifier_path)

    image = load_and_preprocess(image_path)
    prediction = model(image)
    print(prediction)

