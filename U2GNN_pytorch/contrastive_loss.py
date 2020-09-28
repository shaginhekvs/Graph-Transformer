import torch
import torch.nn as nn

class GraphContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, args):
        features = args.features 
        mask=args.mask, 
        labels=None if labels not in vars(args).keys() else args.labels
        dist_fn=None if dist_fn not in vars(args).keys() else args.dist_fn
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

            dot_features = features @ features.T / self.temperature
            
        else:
            dot_features = -dist_fn
            
        logits_max, _ = torch.max(dot_features, dim=1, keepdim=True)
        logits = dot_features - logits_max.detach()  # for numerical stability
        
        # mask-out self-connections
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos.mean()

        return loss
