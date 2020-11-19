import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerSLUSGT(nn.Module):

    def __init__(self, vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout,device, l_att = True ):
        super(TransformerSLUSGT, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.self_attn = nn.MultiheadAttention(self.feature_dim_size, 1, dropout=dropout)
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers #Each U2GNN layer consists of a number of self-attention layers
        self.num_U2GNN_layers = num_U2GNN_layers
        self.vocab_size = vocab_size
        self.sampled_num = sampled_num
        self.device = device
        #
        self.l_att = l_att
        self.u2gnn_layers = torch.nn.ModuleList()
        for _ in range(self.num_U2GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=0.5) # embed_dim must be divisible by num_heads
            self.u2gnn_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
        # Linear function
        self.dropouts = nn.Dropout(dropout)

        
    def forward(self, X_concat, input_x, input_y):
        output_vectors = [] # should test output_vectors = [X_concat]
        input_Tr = F.embedding(input_x, X_concat)
        for layer_idx in range(self.num_U2GNN_layers):
            #
            output_Tr = self.u2gnn_layers[layer_idx](input_Tr)
            output_Tr = torch.split(output_Tr, split_size_or_sections=1, dim=1)[0]
            output_Tr = torch.squeeze(output_Tr, dim=1)
            #new input for next layer
            input_Tr = F.embedding(input_x, output_Tr)
            output_vectors.append(output_Tr)
        if(self.l_att):
            output_vectors = torch.stack(output_vectors,dim=1)
            output_vector = self.self_attn(output_vectors,output_vectors,output_vectors)[0] # attention between different layers.

            output_vector = torch.split(output_vector, split_size_or_sections=1, dim=1)[-1]
            output_vectors = torch.squeeze(output_vector, dim = 1)
            #print(output_vector.shape)
        else:
            output_vectors = torch.cat(output_vectors, dim=1)
        #print(output_vectors.shape)
        #output_vectors = self.dropouts(output_vector)

        #logits = self.ss(output_vectors, input_y)

        return output_vectors

