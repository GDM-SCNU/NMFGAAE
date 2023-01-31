from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
import numpy as np

import networkx as nx
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import community  # pip install  python-louvain


def NMI(pred, labels):
    return metrics.normalized_mutual_info_score(pred, labels)


def AC(pred, labels):
    if type(pred) != np.ndarray:
        pred = pred.numpy()

    if type(labels) != np.ndarray:
        labels = labels.numpy()
    labels = labels.astype(np.int64)
    assert pred.size == labels.size
    D = max(pred.max(), labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred.size):
        w[pred[i], labels[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / pred.size


def Modularity(adj, part, classes_num=None):
    graph = nx.from_numpy_matrix(adj.numpy())
    part = part.tolist()
    index = range(0, len(part))
    dic = zip(index, part)
    part = dict(dic)
    modur = community.modularity(part, graph)
    return modur


def ARI(pred, labels):
    return metrics.adjusted_rand_score(pred, labels)


def kl_div(x, y, reduc='sum'):
    logp_x = F.log_softmax(x, dim=1)
    p_y = F.softmax(y, dim=1)
    kl_val = F.kl_div(logp_x, p_y, reduction=reduc)
    return kl_val


def cosine_simi(a):
    return cosine_similarity(a)


def euclid_dis(x, y):
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return euclidean_distances(x, y)
