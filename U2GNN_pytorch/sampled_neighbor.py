import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


"""LogUniformSampler is taken from https://github.com/rdspring1/PyTorch_GBW_LM"""

def sample_neighbors(graph, node_id, num_to_sample):
    value = node_id
    neighbors_list = [n for n in graph.neighbors(value)]
    input_neighbors = []
    if(neighbors_list):
        input_neighbors.extend(list(np.random.choice(neighbors_list, num_to_sample, replace=True)))
    else:
        pass
    return input_neighbors
class SampledNeighbor(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, device, graph):
        super(SampledNeighbor, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled
        self.device = device
        #
        self.graph = graph
        #
        self.weight = nn.Parameter(torch.Tensor(ntokens, nhid))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        # sample ids according to word distribution - Unique
        #sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
        return self.sampled(inputs, labels)

    """@Dai Quoc Nguyen: Implement the sampled softmax loss function as described in the paper
    On Using Very Large Target Vocabulary for Neural Machine Translation https://www.aclweb.org/anthology/P15-1001/"""
    def sampled(self, inputs, labels):
        assert(inputs.data.get_device() == labels.data.get_device())
        labels_cpu = labels.data.cpu().numpy()
        all_logits = []
        for i,val in enumerate(labels_cpu):
            sample_ids = sample_neighbors(self.graph, val, self.nsampled)
            if(sample_ids):
                sample_ids = Variable(torch.LongTensor(sample_ids)).to(self.device)


                # gather true labels
                true_weights = torch.index_select(self.weight, 0, labels[i:i+1])

                # gather sample ids
                sample_weights = torch.index_select(self.weight, 0, sample_ids)

                # calculate logits
                true_logits = torch.exp(torch.sum(torch.mul(inputs, true_weights), dim=1))
                sample_logits = torch.exp(torch.matmul(inputs, torch.t(sample_weights)))

                logits = -torch.log(true_logits / torch.sum(sample_logits, dim=1))
                all_logits.append(logits)
        logits_cat = torch.stack(all_logits,dim = 1)
        print(logits_cat.shape)
        all_logits = torch.sum(logits_cat,dim = 1)
        print(all_logits.shape)
        return all_logits
