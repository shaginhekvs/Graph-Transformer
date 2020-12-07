import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pytorch_slusgt_UnSup import TransformerSLUSGT
from sampled_softmax import  *

class TransformerMLUSGT(nn.Module):

    def __init__(self, vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout,device, l_att = True , num_graph_layers=1,siamese = False,num_classes = 2 ):
        super(TransformerMLUSGT, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.num_classes = num_classes
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers #Each U2GNN layer consists of a number of self-attention layers
        self.num_U2GNN_layers = num_U2GNN_layers
        self.vocab_size = vocab_size
        self.sampled_num = sampled_num
        self.device = device
        #
        self.l_att = l_att
        self.num_graph_layers = num_graph_layers
        self.slusgt_layers = torch.nn.ModuleList()
        if(siamese):
            model = TransformerSLUSGT(vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout,device, l_att )
            for _ in range(num_graph_layers):
                self.slusgt_layers.append(model)
        else:
            for _ in range(num_graph_layers):
                model = TransformerSLUSGT(vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout,device, l_att )
                
                self.slusgt_layers.append(model)
        
        final_embd_shape = self.feature_dim_size
        
        if(not l_att):
            final_embd_shape = final_embd_shape * num_U2GNN_layers
        self.self_attn = nn.MultiheadAttention(final_embd_shape, 1, dropout=dropout)
         
        self.ss = SampledSoftmax(self.vocab_size, self.sampled_num, final_embd_shape, self.device)
        
        # Linear function
        self.predictions = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(self.num_U2GNN_layers):
            self.predictions.append(nn.Linear(self.feature_dim_size, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, X_concat, input_x, input_y, graph_pool = None , tl = False):
        output_vectors = [] # should test output_vectors = [X_concat]
        for i in range(self.num_graph_layers):
            #print("graph layer {}".format(i))
            output_vectors.append(self.slusgt_layers[i](X_concat[i,:,:],input_x[i,:,:],input_y))
        
        output_vectors = torch.stack(output_vectors,dim=1)#.squeeze(dim = 1)
        output_vector = self.self_attn(output_vectors,output_vectors,output_vectors)[0] # attention between different layers.
        output_vector = torch.split(output_vector, split_size_or_sections=1, dim=1)[-1]
        output_vectors = torch.squeeze(output_vector, dim = 1)
        if(not tl):
            logits = self.ss(output_vectors, input_y)
            return logits
        else:
            prediction_scores = 0
            for i in range(self.num_U2GNN_layers):
                output_Tr = output_vectors[:,i*self.feature_dim_size:(i+1)*self.feature_dim_size]
                graph_embeddings = torch.spmm(graph_pool, output_Tr)
                graph_embeddings = self.dropouts[i](graph_embeddings)
                # Produce the final scores
                prediction_scores += self.predictions[i](graph_embeddings)

            return prediction_scores
            

        

