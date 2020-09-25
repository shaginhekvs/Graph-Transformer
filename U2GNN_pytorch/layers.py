import math
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, Sequence, Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = glorot_init(in_features, out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, device, bias=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_list):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.matmul(x, self.weight)

        source, target = edge_list
        a_input = torch.cat([h[source], h[target]], dim=1)
        e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)

        N = h.size(0)
        attention = -1e20*torch.ones([N, N], device=self.device, requires_grad=True)
        attention[source, target] = e[:, 0]
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h = F.dropout(h, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime
    
    
    
class TransformerEncoderLayerSmaller(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", bias = True):
        super(TransformerEncoderLayerSmaller, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_feedforward, nhead, dropout=dropout)
        #nn.init.xavier_uniform(self.self_attn.weight)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.dropout = nn.Dropout(dropout)
        #self.linear2 = Linear(dim_feedforward, d_model)

        #self.norm1 = LayerNorm(d_model)
        #self.norm2 = LayerNorm(d_model)
        #self.dropout1 = Dropout(dropout)
        #self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        

        if bias:
                self.bias = Parameter(torch.FloatTensor(dim_feedforward))
        else:
                self.register_parameter('bias', None)
        self.reset_parameters()
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerSmaller, self).__setstate__(state)
    
    def reset_parameters(self):
        #nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        #nn.init.xavier_uniform_(self.a.data, gain=1.414)
        

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        src = src #+ self.dropout(src2)
        #src = self.norm1(src)
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.activation(self.linear1(src))
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.dropout(src2)
        #src = src + self.dropout2(src2)
        #src = self.norm2(src)
        if self.bias is not None:
            src = src + self.bias
        return src
    
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))