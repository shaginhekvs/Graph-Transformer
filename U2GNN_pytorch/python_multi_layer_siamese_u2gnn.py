import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TransformerEncoderLayerSmaller
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from .sampled_softmax import  SampledSoftmax
from .sampled_neighbor import SampledNeighbor
from .pytorch_U2GNN_UnSup import TransformerU2GNN
from .contrastive_loss import GraphContrastiveLoss
from .loss_functions import Loss_functions
from .util import Namespace


class TransformerMLU2GNN(nn.Module):

    def __init__(self, vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout, device, sampler_type = 'default', loss_type = 'default', adj_mat = None, single_layer_only = False, ml_model_type = 'siamese', projection_dim = -1, alpha = 0.2):
        super(TransformerMLU2GNN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.device = device
        self.self_attn = nn.MultiheadAttention(self.feature_dim_size, 1, dropout=dropout)
        self.u2gnn_model_per_layer = torch.nn.ModuleList()
        self.adj_mat = adj_mat
        self.alpha = alpha
        self.ml_model_type =  ml_model_type
        self.num_u2gnn_layers = 1 if ml_model_type == 'siamese' else adj_mat.shape[-1]
        self.num_graph_layers = adj_mat.shape[-1]
        self.loss_func = Loss_functions(loss_type)
        
        self.weight = nn.Parameter(torch.Tensor(vocab_size, feature_dim_size))
        self.projection_dim = projection_dim
        self.proj_weight = None
        if(self.projection_dim > 0):
            self.proj_weight = nn.Parameter(torch.FloatTensor(self.feature_dim_size, self.projection_dim))
            self.weight = nn.Parameter(torch.Tensor(vocab_size, self.projection_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(vocab_size, feature_dim_size))
        for i in range(self.num_u2gnn_layers):
            u2gnn_model = TransformerU2GNN(vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout, device, sampler_type, loss_type, adj_mat[i],single_layer_only = False)
            self.u2gnn_model_per_layer.append(u2gnn_model)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if(self.projection_dim > 0):
            nn.init.xavier_uniform_(self.proj_weight.data, gain=1.414)
            
    def ml_loss_func(self,args, logits):
        args_loss = None

        if(args.loss_type == 'gae'):
            args_loss = Namespace(norm = args.norm, logits_list = logits, adj_label = args.adj_label, weight_tensor = args.weight_tensor)
            return args_loss
        elif( args.loss_type == "contrastive"):
            args_loss = Namespace(logits_list = logits, adj_mat = self.adj_mat, device = self.device)
        else:
            raise NotImplementedError('unknown loss {}'.format(args.loss_type))
        return args_loss
    
    def forward(self, X_concat, input_x, input_y=None, args = None):
        if(len(input_x.shape) != 3 ):
            raise ValueError('expected 3d sampled input_x ')
        if(input_x.shape[2] != self.num_graph_layers):
            raise ValueError('there should be a sample for every graph')
            
        logits_all_forwarded = []
        
        
        if(self.ml_model_type == 'siamese'):
            
            for i in range(self.num_graph_layers):
                loss, logits_this = self.u2gnn_model_per_layer[0](X_concat[:,:,i], input_x[:,:,i], input_y)
                
                logits_all_forwarded.append(logits_this)

        
        else:
            
            for i in range(self.num_graph_layers):
                loss, logits_this = self.u2gnn_model_per_layer[i](X_concat[:,:,i], input_x[:,:,i], input_y)
                
                logits_all_forwarded.append(logits_this)
        
        logits_all = torch.stack(logits_all_forwarded,dim=1)
        logits_output = self.self_attn(logits_all,logits_all,logits_all)[0] # attention between different layers    
        output_vector = torch.split(logits_output, split_size_or_sections=1, dim=1)[-1]
        output_vector = torch.squeeze(output_vector, dim = 1)
        if(self.projection_dim > 0):
            output_vector = F.leaky_relu(torch.matmul(output_vector, self.proj_weight), negative_slope=self.alpha)
        output_vector = torch.mul(self.weight, output_vector)    

        
        
        loss_value = self.loss_func(self.ml_loss_func(args,[self.weight]*self.num_graph_layers))   
        
        return loss_value, self.weight.detach()


        
        