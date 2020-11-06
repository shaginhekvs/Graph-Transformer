import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .util import Namespace
class GraphContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.epsilon = torch.tensor(1e-8, dtype = torch.float)

    def forward(self, args):
        features = args.features 
        mask=args.mask
        input_x = args.input_x
        input_samples = args.input_samples
        """
        Args:
            features: embedding matrix of shape (batch_size, latent_dim)
            mask: contrastive mask of shape (batch_size, batch_size)
                  mask_{i,j}=1 if sample j has the same class as sample i. 
                  Can be asymmetric.
            labels: ground truth vector of shape (batch_size,)
        Returns:
            A loss scalar.
        """
        
        #print(input_samples.shape)
        labels=None if 'labels' not in vars(args).keys() else args.labels
        dist_fn=None if 'dist_fn' not in vars(args).keys() else args.dist_fn
        """
        Args:
            features: embedding matrix of shape (batch_size, latent_dim)
            mask: contrastive mask of shape (batch_size, batch_size)
                  mask_{i,j}=1 if sample j has the same class as sample i. 
                  Can be asymmetric.
            labels: ground truth vector of shape (batch_size,)
        Returns:
            A loss scalar.
        """
        cur_label_embeddings = args.output_vector
        #cur_label_embeddings = F.embedding(input_x[:,1], features) 
        #print(cur_label_embeddings.shape)
        
        neigh_embeddings = F.embedding(input_x[:,1:] , features)
        cur_for_neigh = torch.stack([cur_label_embeddings] * neigh_embeddings.shape[1] , dim = 1)
        
        dot_neigh = torch.mul(neigh_embeddings, cur_for_neigh).sum(dim = 2)#.sum(dim = 1) / neigh_embeddings.shape[1]
        
        
        sample_embeddings = F.embedding(input_samples,features)
        cur_for_sample = torch.stack([cur_label_embeddings]* sample_embeddings.shape[1], dim = 1)
        
        #print(cur_for_sample.shape)
        dot_sample = torch.mul(sample_embeddings, cur_for_sample).sum(dim = 2)
        
        dot_features = torch.cat([dot_neigh,dot_sample], dim = 1)
        logits_max, _ = torch.max(dot_features, dim=1, keepdim=True)
        dot_features = dot_features - logits_max.detach()  # for numerical stability
        
        dot_neigh = dot_features[:,:dot_neigh.shape[1]]
        dot_sample = dot_features[:,dot_neigh.shape[1]:]
        
        dot_neigh = dot_neigh.sum(dim = 1) / neigh_embeddings.shape[1]
        dot_sample = torch.log(self.epsilon + torch.exp(dot_sample).sum(dim = 1)/sample_embeddings.shape[1])
        
        #logits = dot_neigh - dot_sample
        logits = torch.log(self.epsilon + torch.exp(dot_neigh)) - dot_sample
        loss_final = - logits.mean()
        
        return loss_final
        '''
        # compute logits
        if dist_fn==None:
            
            batch_size = features.shape[0]
            latent_dim = features.shape[1]

            if len(features.shape) != 2:
                raise ValueError('`features` needs to be a matrix')
            if labels is not None and mask is not None:
                raise ValueError('Cannot define both `labels` and `mask`')
            elif labels is None and mask is None:
                raise ValueError('Must define either `labels` or `mask`')
            elif labels is not None:
                labels = labels.view(-1, 1)
                if labels.shape[0] != batch_size:
                    raise ValueError('Num of labels does not match num of features')
                mask = torch.eq(labels, labels.T).float()
            else:
                mask = mask.float()

            #dot_features = features @ features.T / self.temperature 
            #N X N  matrix product, calculates all pairwise unnormalized dots products
            
        else:
            dot_features = -dist_fn 
            
        #logits_max, _ = torch.max(dot_features, dim=1, keepdim=True)
        #logits = dot_features - logits_max.detach()  # for numerical stability
        losses = []
        for i in range(input_x.shape[0]):
            neighbors = torch.nonzero(mask[i]).flatten()
            cur_node = input_x[i,0].data.numpy()
            #print(cur_node)
            neighbors = input_x[i, 1:]
            feature_i = features[cur_node]
            count_n = neighbors.shape[0]
            numer = []
            for k in neighbors.data.numpy():
                dot_prod = torch.mul(feature_i, features[k])
                #dot_max,_ = torch.max(dot_prod)
                #dot_prod = dor_prod - dot_max.detach()
                numer.append(torch.sum(dot_prod,0,keepdim = True))
            denom = []
            sample_ids = np.random.randint(0,len(features), args.sampled_num)
            for j in sample_ids:#range(len(features)):
                if(i == j):
                    continue   
                
                denom.append(torch.sum(torch.mul(feature_i, features[j]),0,keepdim = True))
            
            numer_vector = torch.cat(numer,dim = 0)
            #numer_max, _ = torch.min(numer_vector ,dim=0, keepdim=True)
            #numer_vector = numer_vector - numer_max.detach()
            
            
            denom_vector = torch.cat(denom,dim = 0)
            #denom_max, _ = torch.max(denom_vector, dim=0, keepdim=True)
            #denom_vector = denom_vector- denom_max.detach()
            
            summed_num = numer_vector.sum(dim = 0,keepdim=True)
            summed_denom = torch.exp(denom_vector).sum(dim = 0,keepdim=True)
            
            losses.append((summed_num -  torch.log(summed_denom)) / count_n)
        
        final_loss =  - torch.cat(losses,dim = 0).mean()
        
        return final_loss
        
        # mask-out self-connections
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        numerator = (mask * log_prob).sum(1)
        #print(numerator)
        denominator = mask.sum(1) +self.epsilon
        #print(denominator)
        mean_log_prob_pos =  numerator / denominator

        # loss
        loss = - mean_log_prob_pos.mean()
        #print(loss)
        return loss
        '''