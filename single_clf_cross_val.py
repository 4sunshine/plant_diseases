from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from dataset import features_labels
import sys


def cross_val(clf, grid, train_features, train_labels, cv=5, n_jobs=8):
    search = GridSearchCV(clf, grid, cv=cv, n_jobs=n_jobs)
    search.fit(train_features, train_labels)
    print(f'Best score {search.best_score_}')
    print(f'Best params {search.best_params_}')
    print(f'Best estimator {search.best_estimator_}')


if __name__ == '__main__':
    # TODO: TEST IT
    features, labels = features_labels(sys.argv[1])

    classifier = MLPClassifier(random_state=1, max_iter=10000)
    grid = {'hidden_layer_sizes': [(60,), (80,), (100,), (120,), (60, 80), (60, 100), (80, 60),
                                   (80, 80), (80, 100), (100, 100), (100, 80), (100, 60)],
            'activation': ['logistic', 'relu']}
    cross_val(classifier, grid, features, labels)

