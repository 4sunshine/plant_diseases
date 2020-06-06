import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


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


root_dir = '/home/sunshine/irishka/0.3v/ds/src'

train_file = os.path.join(root_dir, 'train_avgpooled.pth')
test_file = os.path.join(root_dir, 'test_avgpooled.pth')

train_features, train_labels = features_labels(train_file)
test_features, test_labels = features_labels(test_file)

indices = [0,1,2,3,4,5,6,7,8,11,12,13,14]

train_features = center_data(train_features[:, indices])
train_features = train_features.numpy()

test_features = center_data(test_features[:, indices])
test_features = test_features.numpy()

#pca = PCA(n_components=19, whiten=True)
#pca.fit(train_features)

#x = pca.transform(train_features)
#x_test = pca.transform(test_features)

clf = DecisionTreeClassifier(random_state=0, max_depth=7, splitter='best', criterion='entropy')
clf.fit(train_features, train_labels)

predict = clf.predict(test_features)

correct = [1 for p, t in zip(predict, test_labels) if p==t]
print(sum(correct))
print(np.shape(predict)[0])
acc = 100 * sum(correct) / (predict.shape[0])
print(f'{acc:.2f}%')

# grid = {'max_depth': [6, 7, 8, 9, 10, None], 'splitter': ['best'], 'criterion':
#          ['entropy', 'gini'], 'max_leaf_nodes':[None,3,4,5,6]}
#
# search = GridSearchCV(clf, grid, cv=5, n_jobs=8)
# search.fit(x, train_labels)
# print(search.best_score_)
# print(search.best_params_)
# print(search.best_estimator_)


