# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eval(label, pred):
    acc, f1 = cluster_acc(label, pred)
    nmi = nmi_score(label, pred, average_method='arithmetic')
    ari = ari_score(label, pred)
    ami = adjusted_mutual_info_score(label, pred)
    vms = v_measure_score(label, pred)
    fms = fowlkes_mallows_score(label, pred)

    return acc, f1, nmi, ari, ami, vms, fms


# Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index
def eva_no_labels(z, y_pred):
    z = z.detach().cpu().numpy()
    asw = silhouette_score(z, y_pred)
    chi = calinski_harabasz_score(z, y_pred)
    dbi = davies_bouldin_score(z, y_pred)
    return asw, chi, dbi
