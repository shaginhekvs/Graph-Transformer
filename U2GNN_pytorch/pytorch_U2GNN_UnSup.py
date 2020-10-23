import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TransformerEncoderLayerSmaller
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from .sampled_softmax import  SampledSoftmax
from .sampled_neighbor import SampledNeighbor
from .contrastive_loss import GraphContrastiveLoss
from .util import Namespace

class TransformerU2GNN(nn.Module):

    def __init__(self, vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout, device, sampler_type = 'default', loss_type = 'default', adj_mat = None,single_layer_only = True):
        super(TransformerU2GNN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.self_attn = nn.MultiheadAttention(self.feature_dim_size, 1, dropout=dropout)
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers #Each U2GNN layer consists of a number of self-attention layers
        self.num_U2GNN_layers = num_U2GNN_layers
        self.vocab_size = vocab_size
        self.sampled_num = sampled_num
        self.device = device
        self.single_layer_only = single_layer_only
        self.u2gnn_layers = torch.nn.ModuleList()
        self.em_layers = []
        self.adj_mat = adj_mat
        self.loss_type = loss_type
        if(self.single_layer_only):
            self.weight = nn.Parameter(torch.Tensor(vocab_size, feature_dim_size))
            self.reset_parameters()
        '''
        encoder_layer1 = TransformerEncoderLayerSmaller(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=dropout) # embed_dim must be divisible by num_heads
        self.u2gnn_layers.append(encoder_layer1)
        encoder_layer2 = TransformerEncoderLayerSmaller(d_model=self.ff_hidden_size, nhead=1, dim_feedforward=2, dropout=dropout)
        self.u2gnn_layers.append(encoder_layer2)
        '''
        self.embed_layer = nn.Embedding(self.vocab_size, self.feature_dim_size)
            
        for _ in range(self.num_U2GNN_layers):
            #self.em_layers.append(nn.Embedding(self.vocab_size, self.feature_dim_size))
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=0.5) # embed_dim must be divisible by num_heads
            self.u2gnn_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
        # Linear function
        self.dropouts = nn.Dropout(dropout)
        if(sampler_type == "default"): 
            #self.ss = SampledSoftmax(self.vocab_size, self.sampled_num, self.feature_dim_size*self.num_U2GNN_layers, self.device)
            pass
        if(loss_type == 'contrastive'):
            self.ss = GraphContrastiveLoss()
        
    
    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, X_concat, input_x , input_y, args = None):
        output_vectors = [] # should test output_vectors = [X_concat]
        input_x_t = torch.transpose(input_x, 0, 1)
        #print(input_x_t.shape)
        input_Tr = F.embedding(input_x_t, X_concat)
        for layer_idx in range(self.num_U2GNN_layers):
            #
            output_Tr = self.u2gnn_layers[layer_idx](input_Tr)
            input_Tr = output_Tr
            
            output_Tr_t = torch.transpose(output_Tr, 0, 1)
            #print(output_Tr_t.shape)
            output_Tr = torch.split(output_Tr_t, split_size_or_sections=1, dim=1)[0]
            output_Tr = torch.squeeze(output_Tr, dim=1)
            #new input for next layer
            #input_Tr = F.embedding(input_x, output_Tr)
            
            output_vectors.append(output_Tr)
        
        #output_Tr = torch.split(output_Tr, split_size_or_sections=1, dim=1)[0]
        #output_vectors = torch.squeeze(output_Tr, dim=1)
        
        output_vectors = torch.stack(output_vectors,dim=1)
        output_vectors = torch.transpose(output_vectors,0,1)
        #print("for multi layers")
        #print(output_vectors.shape)
        output_vector = self.self_attn(output_vectors,output_vectors,output_vectors)[0] # attention between different layers.
        
        output_vector = torch.transpose(output_vector,0,1)
        #print(output_vector.shape)
        output_vector = torch.split(output_vector, split_size_or_sections=1, dim=1)[-1]
        output_vector = torch.squeeze(output_vector, dim = 1)
        #output_vectors = output_vectors[-1]
        #output_vector = torch.mul(self.weight, output_vector)
        #output_vector = self.dropouts(output_vector)
        
        if(self.single_layer_only):
            output_vector = torch.mul(self.weight, output_vector)
            output_vector = self.dropouts(output_vector)
            if(self.loss_type == 'default'):
                logits = self.ss(output_vectors, input_y)
            elif(self.loss_type == 'gae'):
                logits = self.weight
            elif(self.loss_type == "contrastive"):
                args_loss = Namespace(features = self.weight, mask = self.adj_mat)
                logits = self.ss(args_loss)
            else:
                raise ValueError('unknown loss_type {}'.format(self.loss_type))
            return logits, self.weight
        else:
            return None, output_vector

