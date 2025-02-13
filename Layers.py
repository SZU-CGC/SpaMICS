"""
    Authors: Guangchang Cai
    Date: 2024-9-22
    Version: 1.0
    Description: This file implements a Graph Autoencoder (GAE) using PyTorch.
"""

import torch.autograd
import torch.nn as nn
import opt


class GNNLayer(nn.Module):
    """
    A single GNN layer that performs a linear transformation of the input
    features followed by matrix multiplication with the adjacency matrix (adj).
    """

    def __init__(self, in_features, out_features, activation=nn.Tanh()):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.act = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights using Xavier initialization.
        """
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        """
        Forward pass through the GNN layer.
        - features: Input node features.
        - adj: Adjacency matrix representing graph structure.
        - apply_activation: Whether to apply the activation function.
        """
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)

        return output


class GAE_encoder(nn.Module):
    """
        Encoder part of the Graph Autoencoder.
        Encodes input features into a latent space representation using multiple GNN layers.
    """
    def __init__(self, gae_n_enc_1, gae_n_enc_2, n_input, n_z, dropout):
        super(GAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, n_z)
        self.dropout = nn.Dropout(dropout)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        """
        Forward pass through the encoder.
        - x: Input features.
        - adj: Adjacency matrix.
        Returns latent space representation and reconstructed adjacency matrix.
        """
        z = self.gnn_1(x, adj, active=True)
        z = self.dropout(z)
        z = self.gnn_2(z, adj, active=True)
        z = self.dropout(z)
        z_igae = self.gnn_3(z, adj, active=False)

        # Reconstruct adjacency matrix from latent space representation
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class GAE_decoder(nn.Module):
    """
    Decoder part of the Graph Autoencoder.
    Reconstructs the input features from the latent space representation using GNN layers.
    """
    def __init__(self, gae_n_dec_1, gae_n_dec_2, n_input, n_z):
        super(GAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(n_z, gae_n_dec_1)
        self.gnn_5 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_6 = GNNLayer(gae_n_dec_2, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        """
        Forward pass through the decoder.
        - z_igae: Latent space representation.
        - adj: Adjacency matrix.
        Returns reconstructed features and reconstructed adjacency matrix.
        """
        z = self.gnn_4(z_igae, adj, active=True)
        z = self.gnn_5(z, adj, active=True)
        z_hat = self.gnn_6(z, adj, active=True)

        # Reconstruct adjacency matrix from decoded features
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class GAE(nn.Module):
    """
    Full Graph Autoencoder (GAE) model that includes both the encoder and decoder.
    """
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_dec_1, gae_n_dec_2, n_input, n_z, dropout):
        super(GAE, self).__init__()
        self.encoder = GAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            n_input=n_input,
            n_z=n_z,
            dropout=dropout)

        self.decoder = GAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            n_input=n_input,
            n_z=n_z)


def build_gae(n_input):
    """
    Helper function to construct a GAE model.
    - n_input: The dimensionality of input features.
    """
    try:
        return GAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_dec_1=opt.args.gae_n_dec_1,
            gae_n_dec_2=opt.args.gae_n_dec_2,
            n_input=n_input,
            n_z=opt.args.gcn_z,
            dropout=opt.args.dropout
        ).to(opt.args.device)
    except AttributeError as e:
        raise ValueError(f"Missing argument in opt.args: {e}")
