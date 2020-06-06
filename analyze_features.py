import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


root_dir = '/home/sunshine/irishka/0.3v/ds/src'

train_file = os.path.join(root_dir, 'train_avgpooled.pth')
test_file = os.path.join(root_dir, 'test_avgpooled.pth')

# ALL GLCM PROPERTIES AND DISTANCES
props = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')
dists = (1, 3, 5)
thetas = (0, 1, 2, 3)  # in units of Pi/4


def build_glcm_indices(start=8):
    i = start
    i_p = {}
    i_d = {}
    i_t = {}
    for p in props:
        if not p in i_p.keys():
            i_p[p] = []
        for d in dists:
            if not d in i_d.keys():
                i_d[d] = []
            for t in thetas:
                if not t in i_t.keys():
                    i_t[t] = []
                i_p[p] += [i]
                i_d[d] += [i]
                i_t[t] += [i]
                i += 1
    return i_p, i_d, i_t


i_p, i_d, i_t = build_glcm_indices()


def indices(prop=None, dist=None, theta=None):
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    if prop in props:
        ind += i_p[prop]
    if dist in dists:
        ind += i_d[dist]
    if theta in thetas:
        ind += i_t[theta]
    return sorted(list(set(ind)))


def features_labels(file):
    dataset = torch.load(file)
    features = dataset['features']
    labels = dataset['labels']
    return features, labels


def center_data(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    data = (data - mean) / std
    return data


max_acc = 0
p_best, d_best, t_best = 0, 0, 0

with tqdm(total=len(props)*len(dists)*len(thetas)) as pbar:
    for p in props + (None, ):
        for d in dists + (None, ):
            for t in thetas + (None, ):
                ind = indices(p, d, t)

                train_features, train_labels = features_labels(train_file)
                test_features, test_labels = features_labels(test_file)

                train_features = center_data(train_features[:, ind])
                train_features = train_features.numpy()

                test_features = center_data(test_features[:, ind])
                test_features = test_features.numpy()

                #pca = PCA(n_components=19, whiten=True)
                #pca.fit(train_features)

                #x = pca.transform(train_features)
                #x_test = pca.transform(test_features)

                clf = DecisionTreeClassifier(random_state=0, max_depth=7, splitter='best', criterion='entropy')
                clf.fit(train_features, train_labels)

                predict = clf.predict(test_features)

                correct = [1 for p, t in zip(predict, test_labels) if p==t]
                acc = 100 * sum(correct) / (predict.shape[0])
                if acc > max_acc:
                    max_acc = acc
                    p_best = p
                    t_best = t
                    d_best = d
                    print(f'Max Acc {max_acc:.2f}')

                pbar.update(1)

print(f'Max accuracy {max_acc:.2f}% obtained with p {p_best}')
print(f'theta {t_best} distance {d_best}')

# grid = {'max_depth': [6, 7, 8, 9, 10, None], 'splitter': ['best'], 'criterion':
#          ['entropy', 'gini'], 'max_leaf_nodes':[None,3,4,5,6]}
#
# search = GridSearchCV(clf, grid, cv=5, n_jobs=8)
# search.fit(x, train_labels)
# print(search.best_score_)
# print(search.best_params_)
# print(search.best_estimator_)

