import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import networkx as nx

from .layers import GraphConvolution 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from .sampled_softmax import  SampledSoftmax
from .sampled_neighbor import SampledNeighbor
from .contrastive_loss import GraphContrastiveLoss

class GCNEncoder(nn.Module):
    """Encoder using GCN layers"""

    def __init__(self, n_feat, n_hid, n_latent, dropout, device):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2_mu = GraphConvolution(n_hid, n_latent)
        self.gc2_sig = GraphConvolution(n_hid, n_latent)
        self.dropout = dropout


    def forward(self, x, adj):
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gc2_mu(x, adj)
        log_sig = self.gc2_sig(x, adj)
        return mu, torch.exp(log_sig)



class TransformerGCN(nn.Module):

    def __init__(self, vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout, device, sampler_type = 'default', graph_obj = None, loss_type = 'default', adj_mat = None):
        super(TransformerGCN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers #Each U2GNN layer consists of a number of self-attention layers
        self.num_U2GNN_layers = num_U2GNN_layers
        self.vocab_size = vocab_size
        self.sampled_num = sampled_num
        self.device = device
        self.loss_type = loss_type
        self.adj_mat = adj_mat
        coo_adj = nx.to_scipy_sparse_matrix(graph_obj).tocoo()
        self.indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64)).to(self.device)
    
        self.gcn_encoder = GCNEncoder(feature_dim_size, ff_hidden_size, 2 ,dropout, self.device)
        self.dropouts = nn.Dropout(dropout)
        if(sampler_type == "default"): 
            self.ss = SampledSoftmax(self.vocab_size, self.sampled_num, self.feature_dim_size*self.num_U2GNN_layers, self.device)
        elif(sampler_type == "neighbor"):
            self.ss = SampledNeighbor(self.vocab_size, self.sampled_num, self.feature_dim_size*self.num_U2GNN_layers, self.device, graph_obj)
        if(loss_type == 'contrastive'):
            self.ss = GraphContrastiveLoss()
    def forward(self, X_concat, input_x, input_y, args= None):
        output_vectors = [] # should test output_vectors = [X_concat]
        output_vectors, _ = self.gcn_encoder(X_concat, args.adj_norm)
        if(self.loss_type == 'default'):
            logits = self.ss(output_vectors, input_y)
        elif(self.loss_type == 'gae'):
            logits = output_vectors
        elif(self.loss_type == "contrastive"):
            logits = self.ss(output_vectors, self.adj_mat)
        else:
            raise ValueError('unknown loss_type {}'.format(self.loss_type))
        return logits, output_vectors

