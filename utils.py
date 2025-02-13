import random
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
import opt
import torch.nn as nn
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn import cluster
from termcolor import colored
from scipy.sparse import coo_matrix, csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def reconstruction_loss(X, A_norm, Z_hat, A_hat):
    loss_w = F.mse_loss(Z_hat, torch.mm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm)
    loss_igae = loss_w + opt.args.alpha_value * loss_a
    return loss_igae


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def construct_graph(count, k=10, mode="connectivity"):
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()

    adj = (adj.T + adj) / 2
    adj_n = norm_adj(adj)

    return adj, adj_n


def refine_adj_spatial(feature_graph, spatial_graph):
    mask = np.logical_and(feature_graph > 0, spatial_graph > 0)
    spatial_graph_refine = np.where(mask, spatial_graph, 0)

    return norm_adj(spatial_graph_refine)


def convert_to_tensor(arrays, device=opt.args.device):
    return [torch.FloatTensor(arr).to(device) for arr in arrays]


class WeightFusion(nn.Module):
    def __init__(self, num_views, feature_dim):
        super(WeightFusion, self).__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim

        initial_weights = torch.ones(num_views) / num_views
        self.weights = nn.Parameter(initial_weights, requires_grad=True)
        # self.weights = nn.Parameter(torch.randn(num_views), requires_grad=True)

    def forward(self, *views):
        normalized_weights = F.softmax(self.weights, dim=0)

        fused_feature = sum(w * v for w, v in zip(normalized_weights, views))

        return fused_feature, normalized_weights


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d=11, alpha=4):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def print_metrics(ACC, F1, NMI, ARI, AMI):
    metrics_str = f"| ACC: {ACC:.4f} | NMI: {NMI:.4f} | ARI: {ARI:.4f} | F1: {F1:.4f} | AMI: {AMI:.4f} |"
    border = "=" * len(metrics_str)

    colored_border = colored(border, 'white', attrs=['bold'])
    colored_metrics = colored(metrics_str, 'red', attrs=['bold'])

    print(colored_border)
    print(colored_metrics)
    print(colored_border)
