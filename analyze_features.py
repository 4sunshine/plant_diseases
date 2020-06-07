import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from utils import powerset, center_data, whiten_data, best_result, train_and_test, build_glcm_indices
from itertools import combinations
from dataset import features_labels
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# ALL GLCM PROPERTIES AND DISTANCES
glcm_params = {'props': ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'),
               'dists': (1, 3, 5),
               'thetas': (0, 1, 2, 3)}  # in units of Pi/4

i_p, i_d, i_t = build_glcm_indices()

#ind = [[0,1,2,3,4,5,6,7]+i_d[d] for d in glcm_params['dists']]
ind = [list(range(80))]
#ind = [[4,5,6,7,21]]
#ind = [[]]
#ind = [[0,1,2,3,4,5,6,7,34,35,59]]

root_dir = '/home/sunshine/irishka/0.3v/ds/src'

train_file = os.path.join(root_dir, 'train_avgpooled.pth')
test_file = os.path.join(root_dir, 'test_avgpooled.pth')
dump_file = os.path.join(root_dir, 'dump_thetas_avgpooled.pth')
dump_file = None

#  ind = list(powerset(list(range(8))+list(range(8, 40))))
#glcms_i = list(range(8, 80))
#four_combs = combinations(glcms_i, 3)
#print('COMPLETED_FOUR COMBS')
#ind = [list(f) for f in four_combs]

# clf = DecisionTreeClassifier(random_state=0, splitter='best', criterion='entropy', max_depth=10)
# clf = SVC(gamma='auto', kernel='linear', C=5, shrinking=True)
# clf = RandomForestClassifier(max_depth=10, random_state=0, criterion='entropy', n_jobs=8, n_estimators=250)
# clf = KNeighborsClassifier(n_neighbors=11, n_jobs=8, metric='euclidean')
clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto',
                    beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=(200,), learning_rate='constant',
                    learning_rate_init=0.001, max_fun=15000, max_iter=10000,
                    momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                    power_t=0.5, random_state=1, shuffle=True, solver='adam',
                    tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)


best_acc, best_params = best_result(train_and_test, dump_file, over_params=ind, clf=clf,
                                    train_file=train_file, test_file=test_file, whiten=True, center=True)

print(f'Best accuracy: {100*best_acc:.2f}%')
print(f'Best params: {best_params}')
print('****************')



#ind = [0,1,2,3,4,5,6,7,24,56,57,58]
#
# train_features, train_labels = features_labels(train_file)
# test_features, test_labels = features_labels(test_file)
#
# train_labels = [t.numpy() for t in train_labels]
# test_labels = [t.numpy() for t in test_labels]
#
# train_features = whiten_data(center_data(train_features))
# train_features = train_features.numpy()
#
# test_features = whiten_data(center_data(test_features))
# test_features = test_features.numpy()
#
# #grid = {'max_depth': [6, 7, 8, 9, 10, None], 'splitter': ['best'], 'criterion':
# #         ['entropy', 'gini']}
#
# #clf = SVC()
# #grid = {'gamma':['auto', 'scale'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
# #        'C':[1,1.5,2,3,4,4.5,5], 'degree':[3,4,5], 'random_state':[0], 'shrinking':[True, False]}
#
# grid = {'hidden_layer_sizes': [(60,), (80,), (100,), (120,), (60, 80), (60, 100), (80, 60),
#                                (80, 80), (80, 100), (100, 100), (100, 80), (100, 60)],
#         'activation': ['logistic', 'relu']}
#
# #clf = DecisionTreeClassifier(random_state=0, splitter='best', criterion='gini', max_depth=10)
#
# #clf.fit(train_features, train_labels)
#
# clf = MLPClassifier(random_state=1, max_iter=10000)
#
# search = GridSearchCV(clf, grid, cv=5, n_jobs=8)
# search.fit(train_features, train_labels)
# print(search.best_score_)
# print(search.best_params_)
# print(search.best_estimator_)
#
