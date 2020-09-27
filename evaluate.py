import os
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import torch

from utils import train_and_test
from dataset import features_labels


def build_classifiers():
    classifiers_names = ('Decision Tree', 'Support Vector Machine', 'Random Forest',
                         'K Nearest Neighbors', '1 Layer MLP')

    classifiers = list()
    classifiers.append(DecisionTreeClassifier(random_state=0, splitter='best', criterion='entropy', max_depth=3))
    classifiers.append(SVC(gamma='auto', kernel='linear', C=5, shrinking=True))
    classifiers.append(
        RandomForestClassifier(max_depth=10, random_state=0, criterion='entropy', n_jobs=8, n_estimators=250))
    classifiers.append(KNeighborsClassifier(n_neighbors=11, n_jobs=8, metric='euclidean'))
    classifiers.append(MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto',
                                     beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                     hidden_layer_sizes=(200,), learning_rate='constant',
                                     learning_rate_init=0.001, max_fun=15000, max_iter=10000,
                                     momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                                     power_t=0.5, random_state=1, shuffle=True, solver='adam',
                                     tol=0.0001, validation_fraction=0.1, verbose=False,
                                     warm_start=False))
    return tuple(classifiers), classifiers_names


def classification_results(classifiers, classifiers_names, train_file, test_file):
    # INDICES [FEATURE [: 68], STAT: [:4], HIST [4: 8], GLCM[8: 68]]
    ind = [list(range(68)), list(range(4)), list(range(8)), list(range(8, 68))]
    properties = ('FEATURE', 'STAT', 'STAT+HIST', 'GLCM')
    data = {'name': [], 'properties': [], 'standard': [], 'precision': [], 'recall': [], 'f-score': []}

    train_data = features_labels(train_file)
    test_data = features_labels(test_file)

    best_f = 0.
    best_classifier = None

    for clf, name in tqdm(zip(classifiers, classifiers_names)):
        for i, prop in enumerate(properties):
            for standard in [True, False]:
                clf, metrics = train_and_test(ind=ind[i], clf=clf, train_data=train_data,
                                              test_data=test_data, center=True, whiten=standard)
                precision, recall, f_score = metrics
                data['name'] += [name]
                data['properties'] += [prop]
                data['standard'] += [standard]
                data['precision'] += precision
                data['recall'] += recall
                data['f-score'] += f_score

                if f_score > best_f:
                    best_classifier = clf

    return data, best_classifier


if __name__ == '__main__':
    # TODO: SYS.ARGV or ARGPARSE for PRIVATE DATA as BELOW
    root_dir = '/home/sunshine/irishka/0.3v/ds/src'

    train_file = os.path.join(root_dir, 'train_data.pth')
    test_file = os.path.join(root_dir, 'test_data.pth')

    classifiers, classifiers_names = build_classifiers()

    data, best_classifier = classification_results(classifiers, classifiers_names, train_file, test_file)

    torch.save(best_classifier, '/home/sunshine/irishka/best_classifier.pth')

    df = pd.DataFrame(data=data)
    df.to_excel(os.path.join(root_dir, 'RESULT.xlsx'))

