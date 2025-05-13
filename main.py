import numpy as np
import torch
import opt
from utils import post_proC, print_metrics, set_seed
from model import SpaMICS
from evaluation import eval
from load_data import load_data
import tqdm


if __name__ == '__main__':

    set_seed(seed=opt.args.seed)

    # Load data
    X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, label, adj_spatial_omics1, adj_spatial_omics2 = load_data()

    opt.args.n_omics1 = X_omics1.shape[1]
    opt.args.n_omics2 = X_omics2.shape[1]
    if opt.args.name == 'Human_tonsil':
        opt.args.n_cluster = 7
        label = None
    elif opt.args.name == 'Human_Breast_Cancer':
        opt.args.n_cluster = 18
        label = None
    else:
        opt.args.n_cluster = len(np.unique(label))

    print("=" * 10 + " Pretraining has begun! " + "=" * 10)

    model = SpaMICS(X_omics2.shape[0]).cuda(opt.args.device)
    optimizer0 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(31), ncols=200)

    for epoch in pbar:
        loss_rec = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2, stage=0)

        pretrain_loss = loss_rec
        optimizer0.zero_grad()
        pretrain_loss.backward()
        optimizer0.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(opt.args.pretrain_epoch + 1), ncols=200)
    for epoch in pbar:
        loss_rec = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2, stage=1)

        pretrain_loss = loss_rec
        optimizer.zero_grad()
        pretrain_loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.train_lr)
    pbar2 = tqdm.tqdm(range(opt.args.epoch + 1), ncols=200)
    for epoch in pbar2:

        loss_rec, loss_self, loss_reg, loss_dis, S = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2,
                                                           adj_spatial_omics1, adj_spatial_omics2, stage=2)

        total_loss = loss_rec + opt.args.lambda_1 * loss_self + opt.args.lambda_2 * loss_reg + opt.args.lambda_3 * loss_dis

        optimizer2.zero_grad()
        total_loss.backward()
        optimizer2.step()

        pbar2.set_postfix({'loss': '{0:1.4f}'.format(total_loss)})
        if epoch % 600 == 0 and epoch != 0:
            S_cpu = S.cpu().detach().numpy()
            pred, _ = post_proC(S_cpu, opt.args.n_cluster)

            if label is not None:
                acc, f1, nmi, ari, ami, _, _ = eval(label, pred)
                print_metrics(acc, f1, nmi, ari, ami)
