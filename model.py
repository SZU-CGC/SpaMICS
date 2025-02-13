"""
    Authors: Guangchang Cai
    Date: 2024-9-22
    Version: 1.0
"""

import torch.autograd
import torch.nn as nn
import opt
from utils import he_init_weights, reconstruction_loss, WeightFusion
import torch.nn.functional as F
from Layers import build_gae


class SpaMICS(nn.Module):
    def __init__(self, num_sample):
        super().__init__()

        """
        A model for spatial domain identification in spatial multi-omics.

        The model has three training stages:
        - Stage 0: Pre-training of GAEs(no fusion).
        - Stage 1: Pre-training of GAEs and fusion layers.
        - Stage 2: Full model training.
    
        Parameters
        ----------
        num_sample : int
            The number of samples in the dataset

        Attributes
        ----------
        gae_feature_omics1 : GAE
            Graph autoencoder for feature representation of omics1(future graph).
        gae_feature_omics2 : GAE
            Graph autoencoder for feature representation of omics2(future graph).
        gae_spatial_omics1 : GAE
            Graph autoencoder for spatial representation of omics1(spatial graph).
        gae_spatial_omics2 : GAE
            Graph autoencoder for spatial representation of omics2(spatial graph).
        D_omics1 : nn.Parameter
            Self-representation matrix for omics1, learnable parameter.
        D_omics2 : nn.Parameter
            Self-representation matrix for omics2, learnable parameter.
        C1 : nn.Parameter
            Low-rank matrix shared across omics, learnable parameter.
        fusion_1 : WeightFusion
            Fusion layer for feature and spatial representations of omics1.
        fusion_2 : WeightFusion
            Fusion layer for feature and spatial representations of omics2.
        beta_1 : float
            Weight factor for reconstruction loss of omics1.
        beta_2 : float
            Weight factor for reconstruction loss of omics2.
        s : nn.Sigmoid
            Sigmoid activation function for adjacency matrix calculation.

        Returns
        -------
        loss_rec : FloatTensor
            The reconstruction loss.
        loss_self : FloatTensor, optional
            The self-expression loss.
        loss_reg : FloatTensor, optional
            The regularization loss for the self-representation matrices.
        loss_dis : FloatTensor, optional
            The discriminative constraint loss between omics.
        S : Tensor
            The combined similarity matrix.
        """

        self.num_sample = num_sample

        # Initialize graph autoencoders (GAE) for spatial multi-omics data
        self.gae_feature_omics1 = build_gae(n_input=opt.args.n_omics1)
        self.gae_feature_omics2 = build_gae(n_input=opt.args.n_omics2)
        self.gae_spatial_omics1 = build_gae(n_input=opt.args.n_omics1)
        self.gae_spatial_omics2 = build_gae(n_input=opt.args.n_omics2)

        # Initialize the private self-representation matrix (D_omics1、D_omics2) for each omics.
        self.D_omics1 = nn.Parameter(1.0e-4 * torch.ones(num_sample, num_sample), requires_grad=True)
        self.D_omics2 = nn.Parameter(1.0e-4 * torch.ones(num_sample, num_sample), requires_grad=True)

        # Initialize the low-rank matrix (C1), shared across omics.
        self.C1 = nn.Parameter(1.0e-4 * torch.ones(num_sample, opt.args.ranks), requires_grad=True)

        # Weight fusion layers for feature-latent and spatial-latent integration
        self.fusion_1 = WeightFusion(opt.args.view, opt.args.gcn_z)
        self.fusion_2 = WeightFusion(opt.args.view, opt.args.gcn_z)

        # Initialize weights using He initialization
        self.apply(he_init_weights)

        # Balance parameter for omics reconstruction loss
        self.beta_1, self.beta_2 = self.compute_weights(opt.args.n_omics1, opt.args.n_omics2)

        self.s = nn.Sigmoid()

    def forward(self, omics_1, omics_2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1, adj_spatial_omics2,
                stage=0):

        """
                The forward pass of the model.
                Stage 0: Pretrain GAEs (no fusion)
                Stage 1: Pretrain GAEs + Fusion layers
                Stage 2: Full model training
        """

        z_omics1_feature, adj_omics1_feature_hat = self.gae_feature_omics1.encoder(omics_1, adj_feature_omics1)
        z_omics2_feature, adj_omics2_feature_hat = self.gae_feature_omics2.encoder(omics_2, adj_feature_omics2)
        z_omics1_spatial, adj_omics1_spatial_hat = self.gae_spatial_omics1.encoder(omics_1, adj_spatial_omics1)
        z_omics2_spatial, adj_omics2_spatial_hat = self.gae_spatial_omics2.encoder(omics_2, adj_spatial_omics2)

        if stage == 0:
            loss_rec = self._reconstruction_loss_only(omics_1, omics_2, adj_feature_omics1, adj_feature_omics2,
                                                      adj_spatial_omics1, adj_spatial_omics2, z_omics1_feature,
                                                      adj_omics1_feature_hat,
                                                      z_omics2_feature, adj_omics2_feature_hat, z_omics1_spatial,
                                                      adj_omics1_spatial_hat, z_omics2_spatial, adj_omics2_spatial_hat)

            return loss_rec

        elif stage == 1:
            H_omics1, _ = self.fusion_1(z_omics1_feature, z_omics1_spatial)
            H_omics2, _ = self.fusion_2(z_omics2_feature, z_omics2_spatial)

            adj_fusion_hat_omics1 = self.s(torch.mm(H_omics1, H_omics1.t()))
            adj_fusion_hat_omics2 = self.s(torch.mm(H_omics2, H_omics2.t()))

            loss_rec = self._reconstruction_loss_only(omics_1, omics_2, adj_feature_omics1, adj_feature_omics2,
                                                      adj_spatial_omics1, adj_spatial_omics2, H_omics1, adj_fusion_hat_omics1, H_omics2,
                                                      adj_fusion_hat_omics2, H_omics1, adj_fusion_hat_omics1, H_omics2, adj_fusion_hat_omics2)
            return loss_rec

        elif stage == 2:
            H_omics1, _ = self.fusion_1(z_omics1_feature, z_omics1_spatial)
            H_omics2, _ = self.fusion_2(z_omics2_feature, z_omics2_spatial)

            C = torch.mm(self.C1, self.C1.T)
            D_omics1 = self.D_omics1 - torch.diag(torch.diag(self.D_omics1))
            D_omics2 = self.D_omics2 - torch.diag(torch.diag(self.D_omics2))

            S_omics1 = D_omics1 + C
            S_omics2 = D_omics2 + C

            # Apply the self-representation to the latent fusion features.
            SH_omics1 = torch.matmul(S_omics1, H_omics1)
            SH_omics2 = torch.matmul(S_omics2, H_omics2)

            adj_fusion_hat_omics1 = self.s(torch.mm(SH_omics1, SH_omics1.t()))
            adj_fusion_hat_omics2 = self.s(torch.mm(SH_omics2, SH_omics2.t()))

            loss_rec = self._reconstruction_loss_only(omics_1, omics_2, adj_feature_omics1, adj_feature_omics2,
                                                      adj_spatial_omics1, adj_spatial_omics2, SH_omics1, adj_fusion_hat_omics1,
                                                      SH_omics2, adj_fusion_hat_omics2, SH_omics1, adj_fusion_hat_omics1, SH_omics2, adj_fusion_hat_omics2)

            loss_reg = self._regularization_loss(C, D_omics1, D_omics2)
            loss_self = self._self_expression_loss(SH_omics1, SH_omics2, H_omics1, H_omics2)
            loss_dis = self._discriminative_constraint(D_omics1, D_omics2)

            S = C + (1 / opt.args.view) * (0.5 * (D_omics1 + D_omics1.T) + 0.5 * (D_omics2 + D_omics2.T))
            return loss_rec, loss_self, loss_reg, loss_dis, S

    def _reconstruction_loss_only(self, omics_1, omics_2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                                  adj_spatial_omics2, z_omics1_feature, a_1, z_omics2_feature, a_2, z_omics1_spatial,
                                  a_3, z_omics2_spatial, a_4):
        """
        Compute the reconstruction loss for all omics.
        """

        X_omics1_feature_hat, A1_hat = self.gae_feature_omics1.decoder(z_omics1_feature, adj_feature_omics1)
        X_omics2_feature_hat, A2_hat = self.gae_feature_omics2.decoder(z_omics2_feature, adj_feature_omics2)
        X_omics1_spatial_hat, A3_hat = self.gae_spatial_omics1.decoder(z_omics1_spatial, adj_spatial_omics1)
        X_omics2_spatial_hat, A4_hat = self.gae_spatial_omics2.decoder(z_omics2_spatial, adj_spatial_omics2)

        loss_rec_omics1 = reconstruction_loss(omics_1, adj_feature_omics1, X_omics1_feature_hat, (A1_hat + a_1) / 2) + \
                          reconstruction_loss(omics_1, adj_spatial_omics1, X_omics1_spatial_hat, (A3_hat + a_3) / 2)
        loss_rec_omics2 = reconstruction_loss(omics_2, adj_feature_omics2, X_omics2_feature_hat, (A2_hat + a_2) / 2) + \
                          reconstruction_loss(omics_2, adj_spatial_omics2, X_omics2_spatial_hat, (A4_hat + a_4) / 2)

        return self.beta_1 * loss_rec_omics1 + self.beta_2 * loss_rec_omics2

    def _regularization_loss(self, C, D_omics1, D_omics2):

        """
        Compute the regularization loss for the shared matrix C and private self-representation matrices (D_omics1, D_omics2).
        """

        loss_reg = (1 / (opt.args.view + 1)) * (F.mse_loss(C,
                                                           torch.zeros(self.num_sample, self.num_sample,
                                                                       requires_grad=True).to(
                                                               opt.args.device), reduction='sum') +
                                                F.mse_loss(D_omics1,
                                                           torch.zeros(self.num_sample, self.num_sample,
                                                                       requires_grad=True).to(
                                                               opt.args.device), reduction='sum') +
                                                F.mse_loss(D_omics2,
                                                           torch.zeros(self.num_sample, self.num_sample,
                                                                       requires_grad=True).to(
                                                               opt.args.device), reduction='sum'))

        return loss_reg

    @staticmethod
    def _self_expression_loss(SZ_1, SZ_2, z_omics1, z_omics2):

        """
        Compute the self-expression loss between the self-represented omics latent features and the original latent features.
        """

        return (1 / opt.args.view) * F.mse_loss(SZ_1, z_omics1) + (1 / opt.args.view) * F.mse_loss(SZ_2, z_omics2)

    @staticmethod
    def _discriminative_constraint(D_omics1, D_omics2):

        """
        Apply a discriminative constraint between the omics-specific self-representation matrices.
        """

        return torch.norm(torch.mul(D_omics1, D_omics2).view(-1), p=1)

    @staticmethod
    def compute_weights(n_omics1, n_omics2):

        """
        Calculate the balance parameters for weighting the reconstruction losses for omics1 and omics2 based on feature dimensions
        """

        denominator = n_omics1 + n_omics2
        if denominator == 0:
            raise ValueError("The sum of n_omics1 and n_omics2 should not be zero.")
        return n_omics2 / denominator, n_omics1 / denominator
