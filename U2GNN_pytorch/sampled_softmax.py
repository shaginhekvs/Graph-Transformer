import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

from log_uniform import LogUniformSampler

"""LogUniformSampler is taken from https://github.com/rdspring1/PyTorch_GBW_LM"""

class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, device):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled
        self.device = device
        #
        print("init log sampler")
        self.sampler = LogUniformSampler(self.ntokens)
        #
        self.all_vocabs = np.array(list(range(ntokens)))
        print("init param")
        self.weight = nn.Parameter(torch.Tensor(ntokens, nhid))
        print("init reset params")
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, input_x,  labels):
        # sample ids according to word distribution - Unique

        #sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
        #all_samples = []
        #sample_values = (labels.data.cpu().numpy(),None,None)#np.random.choice(cur_array,self.nsampled,replace = False) 
        #print("in sample softmax")

        sample_values = (np.random.choice(self.all_vocabs,self.nsampled,replace = False) , None , None)
        return self.sampled(inputs, labels, sample_values)
    
    """@Dai Quoc Nguyen: Implement the sampled softmax loss function as described in the paper
    On Using Very Large Target Vocabulary for Neural Machine Translation https://www.aclweb.org/anthology/P15-1001/"""
    def sampled(self, inputs, labels, sample_values):
        assert(inputs.data.get_device() == labels.data.get_device())

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids)).to(self.device)

        # gather true labels
        true_weights = torch.index_select(self.weight, 0, labels)

        # gather sample ids
        sample_weights = torch.index_select(self.weight, 0, sample_ids)

        # calculate logits
        #true_logits = torch.exp(torch.sum(torch.mul(inputs, true_weights), dim=1)/ ( torch.sum(true_weights.square(),dim=1)* torch.sum(inputs.square(),dim = 1)))
        
        true_logits = torch.exp(torch.sum(torch.mul(inputs, true_weights), dim=1))
        sample_logits = torch.exp(torch.matmul(inputs, torch.t(sample_weights)))

        logits = -torch.log(true_logits * 10 / torch.sum(sample_logits, dim=1) )

        return logits.sum()
'''
    """@Dai Quoc Nguyen: Implement the sampled softmax loss function as described in the paper
    On Using Very Large Target Vocabulary for Neural Machine Translation https://www.aclweb.org/anthology/P15-1001/"""
    def sampled(self, inputs, input_x,labels):
        assert(inputs.data.get_device() == labels.data.get_device())
        
        batch_size, d = inputs.size()
        #sample_ids =  sample_values

        np_labels = labels.data.cpu().numpy()

        # gather true labels
        true_weights = torch.index_select(self.weight, 0, labels)
        np_labels = labels.data.cpu().numpy()
        
        # calculate logits        
        dot_prod = torch.sum(torch.mul(inputs, true_weights), dim=1)
        #max_val,_ = torch.max(dot_prod)
        #dot_prod = (dot_prod - max_val).detach()
        #print(inputs.shape)
        #print(true_weights.shape)
        #print("numerator shape {}".format(dot_prod.shape))
        #true_logits = torch.exp(dot_prod)
        all_losses = []
        #np_input_x = input_x.data.cpu().numpy()
        for i in range(len(labels)):
            # gather sample ids
            cur_set = self.all_vocabs - set([int(np_labels[i])])
            #cur_set = cur_set - set(np_input_x[i].flatten())
            cur_list = list(cur_set)
            #print(cur_list)
            cur_array = np.array(cur_list)
            #cur_array  = np_labels
            #print(cur_set)
            #print(np_labels[i])
            #print(type(list(self.all_vocabs)[0]))
            #print(type(int(np_labels[i])))
            sample_ids = np_labels
            #sample_ids = np.random.choice(cur_array,self.nsampled,replace = False)
            
            sample_ids = Variable(torch.LongTensor(sample_ids)).to(self.device)
            
            sample_weights = torch.index_select(self.weight, 0, sample_ids)
            #print(inputs[i,:].unsqueeze(dim = 0).shape)
            dot_prod2 = torch.matmul(inputs[i,:].unsqueeze(dim = 0), torch.t(sample_weights)).squeeze()
            #print('denominator')
            #print(dot_prod2.shape)
            #max_val2, _ = torch.max(dot_prod2, dim = 0, keepdim = True)
            #dot_prod2 = dot_prod2 - max_val2.detach()
            sample_logits = torch.exp(dot_prod2)
            
            neigh_ids = input_x[i,:].flatten()
            #print(neigh_ids.shape)
            neigh_weights = torch.index_select(self.weight, 0, neigh_ids)
            dot_prod_neighs = torch.matmul(inputs[i,:].unsqueeze(dim = 0), torch.t(neigh_weights)).squeeze().sum()
            
            #all_losses.append( torch.log(torch.exp(dot_prod[i]+dot_prod_neighs/self.ntokens) / (self.ntokens *torch.sum(sample_logits ))))
            all_losses.append( torch.log(torch.exp(dot_prod[i]) / (torch.sum(sample_logits ))))
        
        loss = - torch.stack(all_losses, dim=0).sum(dim=0)
        return   loss

'''