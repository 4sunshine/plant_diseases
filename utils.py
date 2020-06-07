import torch
from tqdm import tqdm
from dataset import features_labels
from sklearn.metrics import accuracy_score
from joblib import delayed, Parallel


# ALL GLCM PROPERTIES AND DISTANCES
glcm_params = {'props': ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'),
               'dists': (1, 3, 5),
               'thetas': (0, 1, 2, 3)}  # in units of Pi/4


def build_glcm_indices(start=8):
    i = start
    i_p = {}
    i_d = {}
    i_t = {}
    for p in glcm_params['props']:
        if not p in i_p.keys():
            i_p[p] = []
        for d in glcm_params['dists']:
            if not d in i_d.keys():
                i_d[d] = []
            for t in glcm_params['thetas']:
                if not t in i_t.keys():
                    i_t[t] = []
                i_p[p] += [i]
                i_d[d] += [i]
                i_t[t] += [i]
                i += 1
    return i_p, i_d, i_t


def nonzero_stats(images):
    """STATISTICS ON VALUES NOT EQUAL TO 0"""
    images = torch.flatten(images)
    n_nonzeros = images.numel() - (images == 0).sum()
    sum = images.sum()
    std = torch.std(images.type(torch.DoubleTensor))
    return n_nonzeros, sum, std


def aggregate_stats(data_loader):
    """COMMON STATS ACROSS ALL IMAGES"""
    s = torch.tensor(0)
    count = torch.tensor(0)
    stds = []

    with tqdm(total=len(data_loader)) as pbar:
        for i, data in enumerate(data_loader):
            cnt, sm, std = nonzero_stats(data['images'])
            count += cnt
            s += sm
            stds += [std]
            pbar.update(1)
    mean = s / (count + 1e-7)
    #std = torch.cat(stds, dim=0)
    std = torch.mean(torch.tensor(stds))
    print(std)
    return mean, std


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    """https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset"""
    x = len(iterable)
    masks = [1 << i for i in range(x)]
    # to yield empty: range(1<<x)
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, iterable) if i & mask]


def center_data(data):
    mean = torch.mean(data, dim=0)
    return data - mean


def whiten_data(data):
    std = torch.std(data, dim=0)
    return data / std


def best_result(f, file=None, over_params=[], **kwargs):
#    results = []
    results = Parallel(n_jobs=8)(delayed(f)(a, **kwargs) for a in tqdm(over_params))
        # for a in over_params:
        #     results += [f(a, **kwargs)]
        #     pbar.update(1)
    max_result = max(results)
    best_arg = over_params[results.index(max_result)]
    if file:
        torch.save({'results': results, 'parameters': over_params}, file)
    return max_result, best_arg


def train_and_test(ind, clf, train_file, test_file, center=True, whiten=True):
    train_features, train_labels = features_labels(train_file)
    test_features, test_labels = features_labels(test_file)
    if ind:
        train_features = train_features[:, ind]
        test_features = test_features[:, ind]
    if center:
        train_features = center_data(train_features)
        test_features = center_data(test_features)
    if whiten:
        train_features = whiten_data(train_features)
        test_features = whiten_data(test_features)

    clf.fit(train_features, train_labels)
    prediction = clf.predict(test_features)
    return accuracy_score(test_labels, prediction)

