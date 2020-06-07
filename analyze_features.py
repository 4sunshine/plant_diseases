import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from utils import powerset, center_data, whiten_data, best_result, train_and_test
from itertools import combinations
from dataset import features_labels


root_dir = '/home/sunshine/irishka/0.3v/ds/src'

train_file = os.path.join(root_dir, 'train_avgpooled.pth')
test_file = os.path.join(root_dir, 'test_avgpooled.pth')
dump_file = os.path.join(root_dir, 'dump_onlyglc3_avgpooled.pth')

#  ind = list(powerset(list(range(8))+list(range(8, 40))))
glcms_i = list(range(8, 80))
four_combs = combinations(glcms_i, 3)
print('COMPLETED_FOUR COMBS')
ind = [list(f) for f in four_combs]

clf = DecisionTreeClassifier(random_state=0, splitter='best', criterion='entropy', max_depth=10)

best_acc, best_params = best_result(train_and_test, dump_file, over_params=ind, clf=clf,
                                    train_file=train_file, test_file=test_file, whiten=False, center=False)


print(f'Best accuracy: {100*best_acc:.2f}%')
print(f'Best params: {best_params}')
print('****************')


#
# ind = [0,1,2,3,4,5,6,7,24,56,57,58]
#
# train_features, train_labels = features_labels(train_file)
# test_features, test_labels = features_labels(test_file)
#
# train_features = center_data(train_features[:, ind])
# train_features = train_features.numpy()
#
# test_features = center_data(test_features[:, ind])
# test_features = test_features.numpy()
#
# grid = {'max_depth': [6, 7, 8, 9, 10, None], 'splitter': ['best'], 'criterion':
#          ['entropy', 'gini']}
#
# clf = DecisionTreeClassifier(random_state=0, splitter='best', criterion='gini', max_depth=10)
# #clf.fit(train_features, train_labels)
#
# search = GridSearchCV(clf, grid, cv=5, n_jobs=8)
# search.fit(train_features, train_labels)
# print(search.best_score_)
# print(search.best_params_)
# print(search.best_estimator_)
#
