import torch
import os
import numpy as np
from dataset import PlantDiseaseDataset
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import precision_recall_fscore_support


def sca(a,b):
    b_norm = np.expand_dims(np.sqrt(np.sum(np.multiply(b,b), axis=1)), axis=-1)
    b = b / b_norm
    b = np.expand_dims(b, axis=0)
    a_norm = np.expand_dims(np.sqrt(np.sum(np.multiply(a,a), axis=2)), axis=-1)
    a = a / a_norm
    print(np.shape(a))
    print(np.shape(b))
    result = np.sum(np.multiply(a_norm, b_norm), axis=-1)
    return result


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

    new_labels = []
    new_labels += [0] * len(label_features[healthy])
    all_healthy = np.array(label_features.pop(healthy))

    hea = all_healthy.transpose(0,2,1)
    print(np.shape(hea))
    mean_h = np.mean(hea, axis=0)
    print('------')
    print(np.shape(mean_h))
    result = [sca(hea, mean_h)]
    print(np.shape(result))
    means = []

    for k, v in label_features.items():
        new_labels += [1]*len(v)
        plants = np.array(v)
        plants = plants.transpose(0,2,1)
        result += [sca(plants, mean_h)]
        print(np.shape(plants))
        means += [np.mean(plants, axis=0).T]

    result = np.concatenate(result, axis=0)
    print(np.shape(result))
    print(set(new_labels))
    print('kekek')
    new_labels = np.array(new_labels)
    new_labels = np.expand_dims(new_labels, axis=1)
    print(np.shape(new_labels))
    data = np.concatenate([result, new_labels], axis=1)
    print(np.shape(data))
    np.random.shuffle(data)
    print(np.shape(data))
    train_features = data[:5000,:-1]
    test_features = data[5000:, :-1]
    train_labels = data[:5000,-1]
    print(np.sum(train_labels))
    test_labels = data[5000:, -1]
    clf = DecisionTreeClassifier(random_state=0, splitter='best', criterion='entropy', max_depth=11)
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    pr, rec, fs, sup = precision_recall_fscore_support(test_labels, pred, zero_division=0, average='binary')
    print(pr, rec, fs)

    return pr, rec, fs

dotter()

def plotter():
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


